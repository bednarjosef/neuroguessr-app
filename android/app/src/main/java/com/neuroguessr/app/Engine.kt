package com.neuroguessr.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.exifinterface.media.ExifInterface
import java.io.File

/** Where the big artefacts live: pushed with `adb push`, not bundled in the APK. */
object Paths {
    fun root(ctx: Context): File = ctx.getExternalFilesDir(null)!!
    fun indexDir(ctx: Context) = File(root(ctx), "index")
    fun modelDir(ctx: Context) = File(root(ctx), "model")

    /**
     * Create the asset directories from inside the app before anything is pushed into them.
     * Directories that `adb` creates are owned by the shell user, and on newer Android the
     * app then cannot traverse them — the files are there and world-readable, but invisible.
     */
    fun ensureDirs(ctx: Context) {
        indexDir(ctx).mkdirs()
        modelDir(ctx).mkdirs()
        File(root(ctx), "testdata").mkdirs()
    }

    fun modelFile(ctx: Context): File {
        val dir = modelDir(ctx)
        // prefer the smallest model present that we have validated
        for (n in listOf("encoder_fp16.onnx", "encoder_fp32.onnx")) {
            val f = File(dir, n)
            if (f.exists()) return f
        }
        return File(dir, "encoder_fp32.onnx")
    }
}

class MissingAssets(val detail: String) : Exception(detail)

/** Loads once, then answers queries. Construction is the slow part (mmap + ORT session). */
class Engine private constructor(
    val cfg: Config,
    private val index: IndexStore,
    private val encoder: Encoder,
    private val retrieval: Retrieval,
    val modelName: String,
    val loadMs: Long,
) : AutoCloseable {

    /** Stages exist so the UI can say what it is doing during the seconds the encoder takes. */
    enum class Stage { PREPARING, ENCODING, SEARCHING, RANKING }

    fun locate(bitmap: Bitmap, onStage: (Stage) -> Unit = {}): Pair<LocateResult, Long> {
        onStage(Stage.ENCODING)
        val t0 = System.currentTimeMillis()
        val q = encoder.encode(bitmap)
        val encMs = System.currentTimeMillis() - t0
        onStage(Stage.SEARCHING)
        val r = retrieval.locate(q)
        onStage(Stage.RANKING)
        return Pair(r, encMs)
    }

    override fun close() {
        retrieval.close(); encoder.close(); index.close()
    }

    companion object {
        fun load(ctx: Context, modelName: String? = null, threads: Int = 0,
                 ep: String = "cpu"): Engine {
            val t0 = System.currentTimeMillis()
            Paths.ensureDirs(ctx)
            val cfg = Config.fromAssets(ctx)
            val dir = Paths.indexDir(ctx)
            if (!dir.isDirectory) throw MissingAssets("index directory not found: $dir")
            val model = if (modelName != null) File(Paths.modelDir(ctx), modelName)
            else Paths.modelFile(ctx)
            if (!model.exists()) {
                val seen = Paths.modelDir(ctx).list()?.joinToString(", ") ?: "unreadable"
                throw MissingAssets("encoder not found: ${model.name}\nmodel dir contains: $seen")
            }
            val index = IndexStore(dir, cfg)
            val nt = if (threads > 0) threads
            else Runtime.getRuntime().availableProcessors().coerceAtMost(8)
            val encoder = Encoder(model, nt, ep)
            val eng = Engine(cfg, index, encoder, Retrieval(index, cfg),
                "${model.name}/${encoder.epUsed}/t$nt", System.currentTimeMillis() - t0)
            Log.i("Engine", "loaded ${eng.modelName} + ${cfg.nRows} rows in ${eng.loadMs} ms")
            return eng
        }
    }
}

/** Decode a photo, honouring EXIF rotation, and cap the long side the way the index did. */
fun decodeOriented(file: File, maxSide: Int = 512): Bitmap {
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    BitmapFactory.decodeFile(file.absolutePath, bounds)
    val opts = BitmapFactory.Options()
    var s = 1
    while (maxOf(bounds.outWidth, bounds.outHeight) / (s * 2) >= maxSide) s *= 2
    opts.inSampleSize = s
    var bm = BitmapFactory.decodeFile(file.absolutePath, opts)
        ?: throw IllegalArgumentException("cannot decode ${file.name}")
    val rot = when (ExifInterface(file.absolutePath)
        .getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)) {
        ExifInterface.ORIENTATION_ROTATE_90 -> 90f
        ExifInterface.ORIENTATION_ROTATE_180 -> 180f
        ExifInterface.ORIENTATION_ROTATE_270 -> 270f
        else -> 0f
    }
    if (rot != 0f) {
        val mtx = android.graphics.Matrix().apply { postRotate(rot) }
        bm = Bitmap.createBitmap(bm, 0, 0, bm.width, bm.height, mtx, true)
    }
    val long = maxOf(bm.width, bm.height)
    if (long > maxSide) {
        val f = maxSide.toFloat() / long
        bm = Bitmap.createScaledBitmap(bm, Math.round(bm.width * f), Math.round(bm.height * f), true)
    }
    return bm
}
