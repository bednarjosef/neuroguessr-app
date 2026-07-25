package com.neuroguessr.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.util.Log
import java.io.File
import java.nio.FloatBuffer

/** One forward pass: the descriptor spaces the blend scores in, plus the gate's logits. */
class Query(
    val cls: FloatArray,
    val logits: FloatArray,
    val spaces: Map<String, FloatArray>,
)

class Encoder(modelFile: File, threads: Int, ep: String = "cpu") : AutoCloseable {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    val inputName: String
    val outputNames: List<String>
    val epUsed: String

    init {
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(threads)
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        var used = "cpu"
        try {
            when (ep.lowercase()) {
                "nnapi" -> { opts.addNnapi(); used = "nnapi" }
                "xnnpack" -> { opts.addXnnpack(mapOf("intra_op_num_threads" to threads.toString())); used = "xnnpack" }
            }
        } catch (t: Throwable) {
            Log.w("Encoder", "execution provider '$ep' unavailable, falling back to cpu: ${t.message}")
        }
        epUsed = used
        session = env.createSession(modelFile.absolutePath, opts)
        inputName = session.inputNames.first()
        outputNames = session.outputNames.toList()
    }

    /** Milliseconds spent resizing/normalising the last image, separate from inference. */
    var lastPrepMs: Long = 0
        private set
    var lastInferMs: Long = 0
        private set

    fun encode(bitmap: Bitmap): Query {
        val tPrep = System.currentTimeMillis()
        val chw = Preprocess.toTensor(bitmap)
        lastPrepMs = System.currentTimeMillis() - tPrep
        val tInfer = System.currentTimeMillis()
        OnnxTensor.createTensor(
            env, FloatBuffer.wrap(chw),
            longArrayOf(1, 3, Preprocess.SIZE.toLong(), Preprocess.SIZE.toLong())
        ).use { input ->
            session.run(mapOf(inputName to input)).use { res ->
                fun get(name: String): FloatArray =
                    (res.get(name).get().value as Array<FloatArray>)[0]
                val spaces = HashMap<String, FloatArray>()
                for (n in outputNames) {
                    if (n.startsWith("q_")) spaces[n.removePrefix("q_")] = get(n)
                }
                lastInferMs = System.currentTimeMillis() - tInfer
                Log.i("Encoder", "prep ${lastPrepMs}ms infer ${lastInferMs}ms " +
                        "(input ${bitmap.width}x${bitmap.height})")
                return Query(get("cls"), get("logits"), spaces)
            }
        }
    }

    override fun close() { session.close() }
}
