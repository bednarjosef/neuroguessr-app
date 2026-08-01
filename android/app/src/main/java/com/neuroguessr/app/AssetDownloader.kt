package com.neuroguessr.app

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.job
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * The one-time fetch of the app's brain: the encoder and the packed index, ~2.2 GB.
 *
 * The manifest is baked into the APK, not fetched, so an APK can only ever download the exact
 * asset build it was verified against — the two cannot drift. Files land where `adb push`
 * used to put them, so a phone provisioned over USB downloads nothing.
 *
 * Every file arrives through a `.part` neighbour with HTTP range resume, is hashed as it
 * streams, and is renamed into place only once its SHA-256 matches the manifest. A download
 * interrupted at 1.9 GB restarts with 1.9 GB already on disk.
 */
data class RemoteFile(val path: String, val bytes: Long, val sha256: String)

class AssetManifest(val baseUrl: String, val totalBytes: Long, val files: List<RemoteFile>) {
    companion object {
        fun fromAssets(ctx: Context): AssetManifest {
            val o = JSONObject(
                ctx.assets.open("download_manifest.json").readBytes().decodeToString()
            )
            val arr = o.getJSONArray("files")
            return AssetManifest(
                o.getString("base_url"), o.getLong("total_bytes"),
                (0 until arr.length()).map { i ->
                    val f = arr.getJSONObject(i)
                    RemoteFile(f.getString("path"), f.getLong("bytes"), f.getString("sha256"))
                }
            )
        }
    }
}

data class FetchProgress(
    val doneBytes: Long,
    val totalBytes: Long,
    val fileIndex: Int,
    val fileCount: Int,
    val fileName: String,
    val bytesPerSec: Double,
)

object AssetDownloader {
    private const val TAG = "AssetDownloader"
    private const val SLACK = 300L * 1024 * 1024   // breathing room beyond the assets themselves

    /** True when every file the manifest promises is already present at its full size. */
    fun complete(ctx: Context, m: AssetManifest): Boolean =
        m.files.all { File(Paths.root(ctx), it.path).length() == it.bytes }

    suspend fun run(ctx: Context, m: AssetManifest, onProgress: (FetchProgress) -> Unit) =
        withContext(Dispatchers.IO) {
            val job = coroutineContext.job
            val root = Paths.root(ctx)
            val todo = m.files.filter { File(root, it.path).length() != it.bytes }
            var base = m.totalBytes - todo.sumOf { it.bytes }

            val needed = todo.sumOf { it.bytes } -
                    todo.sumOf { File(root, "${it.path}.part").length() }
            if (root.usableSpace < needed + SLACK) throw IOException(
                "Not enough storage: needs %.1f GB free, only %.1f GB available.".format(
                    (needed + SLACK) / 1e9, root.usableSpace / 1e9
                )
            )

            var speed = 0.0
            var lastPos = 0L
            var winBytes = 0L
            var winT0 = System.currentTimeMillis()
            var lastEmit = 0L
            for ((i, rf) in todo.withIndex()) {
                lastPos = 0
                fun report(pos: Long, force: Boolean) {
                    job.ensureActive()
                    if (pos > lastPos) winBytes += pos - lastPos
                    lastPos = pos
                    val now = System.currentTimeMillis()
                    if (now - winT0 > 500) {
                        val inst = winBytes * 1000.0 / (now - winT0)
                        speed = if (speed == 0.0) inst else 0.75 * speed + 0.25 * inst
                        winBytes = 0; winT0 = now
                    }
                    if (force || now - lastEmit > 150) {
                        lastEmit = now
                        onProgress(FetchProgress(
                            base + pos, m.totalBytes, i + 1, todo.size, rf.path, speed
                        ))
                    }
                }

                var attempt = 0
                while (true) {
                    // the first position of an attempt is bytes already on disk, not bytes
                    // that just travelled — it must not be credited to the speed estimate
                    var first = true
                    try {
                        fetchOne(m.baseUrl + rf.path, rf, root) { pos ->
                            if (first) { lastPos = pos; first = false }
                            report(pos, false)
                        }
                        break
                    } catch (e: IOException) {
                        job.ensureActive()
                        if (++attempt >= 3) throw e
                        Log.w(TAG, "retrying ${rf.path}: ${e.message}")
                        Thread.sleep(2000L * attempt)   // the .part survives; retry resumes it
                    }
                }
                base += rf.bytes
                report(0, force = true)
            }
        }

    /** Download one file into `path.part` (resuming, hashing) and rename it into place. */
    private fun fetchOne(url: String, rf: RemoteFile, root: File, onPos: (Long) -> Unit) {
        val final = File(root, rf.path)
        val part = File(root, "${rf.path}.part")
        part.parentFile?.mkdirs()
        if (final.exists()) final.delete()   // wrong size: stale build or interrupted push

        var offset = part.length()
        if (offset > rf.bytes) { part.delete(); offset = 0 }
        if (offset == rf.bytes) {
            // crashed between the last byte and the rename; verify what's here, no network
            onPos(offset)
            val digest = MessageDigest.getInstance("SHA-256")
            hashInto(digest, part, offset, rf)
            seal(digest, part, final, rf)
            return
        }
        val conn = open(url, offset)
        try {
            if (conn.responseCode == HttpURLConnection.HTTP_OK && offset > 0) {
                offset = 0   // server ignored the Range header; start over
            }
            val digest = MessageDigest.getInstance("SHA-256")
            // the resumed prefix has to pass through the hash too
            if (offset > 0) hashInto(digest, part, offset, rf)
            RandomAccessFile(part, "rw").use { out ->
                out.setLength(offset)
                out.seek(offset)
                var pos = offset
                onPos(pos)
                conn.inputStream.use { s ->
                    val buf = ByteArray(1 shl 18)
                    while (true) {
                        val n = s.read(buf)
                        if (n < 0) break
                        out.write(buf, 0, n)
                        digest.update(buf, 0, n)
                        pos += n
                        onPos(pos)
                    }
                }
                if (pos != rf.bytes) throw IOException(
                    "size mismatch for ${rf.path}: got $pos, wanted ${rf.bytes}"
                )
            }
            seal(digest, part, final, rf)
        } finally {
            conn.disconnect()
        }
    }

    /** Stream the first `count` bytes of `f` through the digest. */
    private fun hashInto(digest: MessageDigest, f: File, count: Long, rf: RemoteFile) {
        f.inputStream().use { s ->
            val buf = ByteArray(1 shl 20)
            var left = count
            while (left > 0) {
                val n = s.read(buf, 0, minOf(buf.size.toLong(), left).toInt())
                if (n <= 0) throw IOException("short read hashing resumed ${rf.path}")
                digest.update(buf, 0, n); left -= n
            }
        }
    }

    /** The only way a file becomes real: its hash matches the manifest, then it is renamed. */
    private fun seal(digest: MessageDigest, part: File, final: File, rf: RemoteFile) {
        val hex = digest.digest().joinToString("") { "%02x".format(it) }
        if (hex != rf.sha256) {
            part.delete()
            throw IOException("checksum mismatch for ${rf.path}; will re-download")
        }
        if (!part.renameTo(final)) throw IOException("cannot move ${rf.path} into place")
    }

    /**
     * Open a connection, following redirects by hand so the Range header provably survives
     * the hop onto the CDN host.
     */
    private fun open(url: String, offset: Long): HttpURLConnection {
        var u = url
        repeat(6) {
            val c = (URL(u).openConnection() as HttpURLConnection).apply {
                instanceFollowRedirects = false
                connectTimeout = 15_000
                readTimeout = 30_000
                if (offset > 0) setRequestProperty("Range", "bytes=$offset-")
            }
            when (c.responseCode) {
                HttpURLConnection.HTTP_OK, HttpURLConnection.HTTP_PARTIAL -> return c
                in 300..399 -> {
                    val loc = c.getHeaderField("Location")
                        ?: throw IOException("redirect without Location from $u")
                    c.disconnect()
                    u = URL(URL(u), loc).toString()
                }
                else -> throw IOException("HTTP ${c.responseCode} from $u")
            }
        }
        throw IOException("too many redirects for $url")
    }
}
