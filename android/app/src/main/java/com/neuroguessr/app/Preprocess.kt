package com.neuroguessr.app

import android.graphics.Bitmap
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/**
 * Pillow-compatible BICUBIC resize.
 *
 * The index was embedded from images resized with `PIL.Image.resize((384,384), BICUBIC)`.
 * Android's own `Bitmap.createScaledBitmap` is bilinear, and the resulting descriptor drifts
 * from the index — which is a retrieval error, not a cosmetic one. So this reimplements
 * Pillow's separable resampler: per-output-pixel weights from the Catmull-Rom cubic (a = -0.5),
 * support widened by the downscale factor, normalised to sum 1, horizontal pass then vertical,
 * with the same round-to-uint8 between passes.
 */
object Preprocess {
    const val SIZE = 384
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    private fun cubic(xIn: Double): Double {
        val a = -0.5
        val x = if (xIn < 0) -xIn else xIn
        return when {
            x < 1.0 -> ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
            x < 2.0 -> (((x - 5.0) * x + 8.0) * x - 4.0) * a
            else -> 0.0
        }
    }

    private class Coeffs(val bounds: IntArray, val weights: DoubleArray, val kmax: Int)

    /** Pillow's precompute_coeffs for one axis. */
    private fun coeffs(inSize: Int, outSize: Int): Coeffs {
        val scale = inSize.toDouble() / outSize
        val filterScale = max(1.0, scale)
        val support = 2.0 * filterScale
        val kmax = ceil(support).toInt() * 2 + 1
        val bounds = IntArray(outSize * 2)
        val w = DoubleArray(outSize * kmax)
        for (xx in 0 until outSize) {
            val center = (xx + 0.5) * scale
            val ss = 1.0 / filterScale
            var xmin = floor(center - support + 0.5).toInt()
            if (xmin < 0) xmin = 0
            var xmax = floor(center + support + 0.5).toInt()
            if (xmax > inSize) xmax = inSize
            xmax -= xmin
            var ww = 0.0
            for (x in 0 until xmax) {
                val v = cubic((x + xmin - center + 0.5) * ss)
                w[xx * kmax + x] = v
                ww += v
            }
            if (ww != 0.0) for (x in 0 until xmax) w[xx * kmax + x] /= ww
            bounds[xx * 2] = xmin
            bounds[xx * 2 + 1] = xmax
        }
        return Coeffs(bounds, w, kmax)
    }

    private fun clamp8(v: Double): Int = min(255.0, max(0.0, floor(v + 0.5))).toInt()

    /**
     * Resize to [SIZE]x[SIZE] (aspect is intentionally NOT preserved — the training pipeline
     * squashed to a square too) and return a normalised NCHW float tensor.
     */
    fun toTensor(src: Bitmap): FloatArray {
        val w = src.width
        val h = src.height
        val px = IntArray(w * h)
        src.getPixels(px, 0, w, 0, 0, w, h)

        // horizontal pass: w x h -> SIZE x h, per channel, rounded back to 8-bit like Pillow
        val cx = coeffs(w, SIZE)
        val tmp = ByteArray(SIZE * h * 3)
        for (y in 0 until h) {
            val rowOff = y * w
            for (xx in 0 until SIZE) {
                val xmin = cx.bounds[xx * 2]
                val xlen = cx.bounds[xx * 2 + 1]
                var r = 0.0; var g = 0.0; var b = 0.0
                for (k in 0 until xlen) {
                    val wt = cx.weights[xx * cx.kmax + k]
                    val p = px[rowOff + xmin + k]
                    r += wt * ((p shr 16) and 0xFF)
                    g += wt * ((p shr 8) and 0xFF)
                    b += wt * (p and 0xFF)
                }
                val o = (y * SIZE + xx) * 3
                tmp[o] = clamp8(r).toByte()
                tmp[o + 1] = clamp8(g).toByte()
                tmp[o + 2] = clamp8(b).toByte()
            }
        }

        // vertical pass: SIZE x h -> SIZE x SIZE, straight into the normalised CHW tensor
        val cy = coeffs(h, SIZE)
        val out = FloatArray(3 * SIZE * SIZE)
        val plane = SIZE * SIZE
        for (yy in 0 until SIZE) {
            val ymin = cy.bounds[yy * 2]
            val ylen = cy.bounds[yy * 2 + 1]
            for (x in 0 until SIZE) {
                var r = 0.0; var g = 0.0; var b = 0.0
                for (k in 0 until ylen) {
                    val wt = cy.weights[yy * cy.kmax + k]
                    val o = ((ymin + k) * SIZE + x) * 3
                    r += wt * (tmp[o].toInt() and 0xFF)
                    g += wt * (tmp[o + 1].toInt() and 0xFF)
                    b += wt * (tmp[o + 2].toInt() and 0xFF)
                }
                val i = yy * SIZE + x
                out[i] = ((clamp8(r) / 255.0f) - MEAN[0]) / STD[0]
                out[plane + i] = ((clamp8(g) / 255.0f) - MEAN[1]) / STD[1]
                out[2 * plane + i] = ((clamp8(b) / 255.0f) - MEAN[2]) / STD[2]
            }
        }
        return out
    }
}
