package com.neuroguessr.app

import android.content.Context
import org.json.JSONObject

/** The champion recipe's constants, exported alongside the index so the two can never drift. */
class Config(private val json: JSONObject) {
    val nRows = json.getInt("n_rows")
    val nCells = json.getInt("n_cells")
    val dim = json.getInt("dim")
    val viewsPerLocation = json.optInt("views_per_location", 1)
    val spaces: List<String> = json.getJSONArray("spaces").let { a ->
        (0 until a.length()).map { a.getString(it) }
    }
    val cslsSpaces: List<String> = json.getJSONArray("csls_spaces").let { a ->
        (0 until a.length()).map { a.getString(it) }
    }
    /** Scoring spaces and their blend weights (does NOT include "cls", which only feeds E7). */
    val blend: Map<String, Double> = json.getJSONObject("blend").let { o ->
        o.keys().asSequence().associateWith { o.getDouble(it) }
    }
    val gateMass = json.getDouble("gate_mass")
    val gateCap = json.getInt("gate_cap")
    val priorLam = json.getDouble("prior_lam")
    val topK = json.getInt("top_k")
    val e7Lam = json.getDouble("e7_lam")
    val scoreSd = json.getDouble("score_sd")

    val e7W: DoubleArray = json.getJSONArray("e7_w").let { a ->
        DoubleArray(a.length()) { a.getDouble(it) }
    }
    val e7B = json.getDouble("e7_b")
    val e7Mu: DoubleArray = json.getJSONArray("e7_feat_mu").let { a ->
        DoubleArray(a.length()) { a.getDouble(it) }
    }
    val e7Sd: DoubleArray = json.getJSONArray("e7_feat_sd").let { a ->
        DoubleArray(a.length()) { a.getDouble(it) }
    }

    // Confidence model. Feature order is fixed by export/fit_confidence.py:
    // agree10, agree25, agree100, top_sim, margin, entropy, log_cand
    private fun arr(key: String, fallback: DoubleArray) =
        json.optJSONArray(key)?.let { a -> DoubleArray(a.length()) { a.getDouble(it) } } ?: fallback

    val confW = arr("conf_w", DoubleArray(0))
    val confB = json.optDouble("conf_b", 0.0)
    val confMu = arr("conf_mu", DoubleArray(0))
    val confSd = arr("conf_sd", DoubleArray(0))
    val confHitKm = json.optDouble("conf_hit_km", 25.0)
    val hasConfidence get() = confW.isNotEmpty()

    companion object {
        fun fromAssets(ctx: Context): Config =
            ctx.assets.open("config.json").bufferedReader().use {
                Config(JSONObject(it.readText()))
            }
    }
}
