package com.neuroguessr.app

import android.util.Log
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * A read-only mmap of a flat file that may exceed the 2 GB limit of a single MappedByteBuffer.
 * The index is written row-major with a fixed row stride, and every read the retrieval does is
 * a contiguous run of rows, so segment boundaries are handled by splitting the copy.
 */
class SegmentedMap(file: File) : AutoCloseable {
    private val raf = RandomAccessFile(file, "r")
    val size: Long = raf.length()
    private val segShift = 30                       // 1 GiB segments
    private val segSize = 1L shl segShift
    private val segs: Array<ByteBuffer>

    init {
        val n = ((size + segSize - 1) / segSize).toInt().coerceAtLeast(1)
        segs = Array(n) { i ->
            val off = i * segSize
            val len = minOf(segSize, size - off)
            raf.channel.map(FileChannel.MapMode.READ_ONLY, off, len).order(ByteOrder.LITTLE_ENDIAN)
        }
    }

    /** Copy [len] bytes starting at absolute [off] into [dst] at [dstOff]. */
    fun copyTo(off: Long, dst: ByteArray, dstOff: Int, len: Int) {
        var remaining = len
        var src = off
        var d = dstOff
        while (remaining > 0) {
            val si = (src ushr segShift).toInt()
            val within = (src and (segSize - 1)).toInt()
            val take = minOf(remaining.toLong(), segSize - within).toInt()
            val b = segs[si].duplicate()
            b.position(within)
            b.get(dst, d, take)
            src += take; d += take; remaining -= take
        }
    }

    override fun close() { raf.close() }
}

/** Everything the retrieval reads from disk, mapped once at startup. */
class IndexStore(dir: File, val cfg: Config) : AutoCloseable {
    val nRows = cfg.nRows
    val dim = cfg.dim

    val latlon: FloatArray        // 2*nRows
    val cellId: ShortArray        // nRows (uint16 stored in signed short)
    val cellOffsets: IntArray     // nCells+1
    val csls: FloatArray          // 3*nRows
    private val vec = HashMap<String, SegmentedMap>()
    private val scale = HashMap<String, FloatArray>()

    init {
        fun readFloats(name: String, n: Int): FloatArray {
            val f = File(dir, name)
            require(f.exists()) { "missing index file: ${f.absolutePath}" }
            val out = FloatArray(n)
            RandomAccessFile(f, "r").use { r ->
                val b = r.channel.map(FileChannel.MapMode.READ_ONLY, 0, f.length())
                    .order(ByteOrder.LITTLE_ENDIAN)
                b.asFloatBuffer().get(out)
            }
            return out
        }
        latlon = readFloats("latlon.f32", nRows * 2)
        csls = readFloats("csls_r.f32", nRows * 3)
        cellId = ShortArray(nRows).also { out ->
            val f = File(dir, "cell_id.u16")
            RandomAccessFile(f, "r").use { r ->
                r.channel.map(FileChannel.MapMode.READ_ONLY, 0, f.length())
                    .order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(out)
            }
        }
        cellOffsets = IntArray(cfg.nCells + 1).also { out ->
            val f = File(dir, "cell_offsets.u32")
            RandomAccessFile(f, "r").use { r ->
                r.channel.map(FileChannel.MapMode.READ_ONLY, 0, f.length())
                    .order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(out)
            }
        }
        for (s in cfg.spaces) {
            vec[s] = SegmentedMap(File(dir, "$s.i8"))
            scale[s] = readFloats("$s.scale.f32", nRows)
            require(vec[s]!!.size == nRows.toLong() * dim) {
                "$s.i8 is ${vec[s]!!.size} bytes, expected ${nRows.toLong() * dim}"
            }
        }
        Log.i("IndexStore", "mapped $nRows rows x $dim dims, ${cfg.spaces.size} spaces")
    }

    fun cellOf(row: Int): Int = cellId[row].toInt() and 0xFFFF
    fun latOf(row: Int): Double = latlon[row * 2].toDouble()
    fun lonOf(row: Int): Double = latlon[row * 2 + 1].toDouble()
    fun scaleOf(space: String): FloatArray = scale[space]!!

    /** Copy the int8 descriptors of rows [from, from+count) of [space] into [dst]. */
    fun readRows(space: String, from: Int, count: Int, dst: ByteArray) {
        vec[space]!!.copyTo(from.toLong() * dim, dst, 0, count * dim)
    }

    /** Copy the int8 descriptor of a single row into [dst] at [dstOff]. */
    fun readRow(space: String, row: Int, dst: ByteArray, dstOff: Int) {
        vec[space]!!.copyTo(row.toLong() * dim, dst, dstOff, dim)
    }

    override fun close() { vec.values.forEach { it.close() } }
}
