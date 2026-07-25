package com.neuroguessr.app

import java.util.concurrent.Callable
import java.util.concurrent.Executors
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

class Candidate(val row: Int, val lat: Double, val lon: Double, val score: Double, var rerank: Double)

class LocateResult(
    val lat: Double,
    val lon: Double,
    val candidates: List<Candidate>,
    val gateCells: Int,
    val nCandidates: Int,
    val msGate: Long,
    val msScore: Long,
    val msRerank: Long,
    /** Calibrated P(this answer is within [Config.confHitKm] of the truth), 0..1. */
    val confidence: Double,
)

/**
 * The champion recipe, on device.
 *
 * gate (classifier posterior mass) -> blended int8 dot products with CSLS hubness correction
 * and a log-prior term -> top-K -> E7 learned reranker -> the winning candidate's coordinates.
 * Every constant comes from config.json, which is written by the same script that packs the
 * index, so the recipe and the vectors can never disagree.
 */
class Retrieval(private val idx: IndexStore, private val cfg: Config) : AutoCloseable {
    private val pool = Executors.newFixedThreadPool(
        Runtime.getRuntime().availableProcessors().coerceAtMost(8)
    )
    private val scoringSpaces = cfg.blend.keys.toList()

    fun locate(q: Query): LocateResult {
        val t0 = System.currentTimeMillis()

        // ---- gate: smallest set of cells holding `gateMass` of the posterior ----------
        val probs = softmaxInPlace(q.logits.copyOf())
        val order = (0 until cfg.nCells).sortedByDescending { probs[it] }
        var mass = 0.0
        var m = 0
        while (m < order.size && m < cfg.gateCap) {
            mass += probs[order[m]]
            m++
            if (mass >= cfg.gateMass) break
        }
        val cells = IntArray(m) { order[it] }

        // candidate rows are contiguous per cell because the index is written cell-sorted
        val cellStart = IntArray(m)
        var total = 0
        for (i in 0 until m) {
            cellStart[i] = total
            total += idx.cellOffsets[cells[i] + 1] - idx.cellOffsets[cells[i]]
        }
        val t1 = System.currentTimeMillis()
        if (total == 0) return LocateResult(0.0, 0.0, emptyList(), m, 0, t1 - t0, 0, 0, 0.0)

        // ---- blended scoring over the gated pool -------------------------------------
        val scores = DoubleArray(total)
        val rows = IntArray(total)
        val nThreads = (pool as java.util.concurrent.ThreadPoolExecutor).maximumPoolSize
        val chunk = (m + nThreads - 1) / nThreads
        val tasks = (0 until nThreads).map { t ->
            Callable {
                val lo = t * chunk
                val hi = minOf(m, lo + chunk)
                if (lo >= hi) return@Callable
                var maxRun = 0
                for (i in lo until hi) {
                    val c = cells[i]
                    maxRun = maxOf(maxRun, idx.cellOffsets[c + 1] - idx.cellOffsets[c])
                }
                val buf = ByteArray(maxRun * cfg.dim)
                for (i in lo until hi) {
                    val c = cells[i]
                    val from = idx.cellOffsets[c]
                    val cnt = idx.cellOffsets[c + 1] - from
                    if (cnt == 0) continue
                    val base = cellStart[i]
                    val lp = cfg.priorLam * ln(probs[c].toDouble().coerceAtLeast(1e-12))
                    for (k in 0 until cnt) rows[base + k] = from + k
                    java.util.Arrays.fill(scores, base, base + cnt, lp)
                    for (sp in scoringSpaces) {
                        val w = cfg.blend[sp]!!
                        val qv = q.spaces[sp] ?: continue
                        val sc = idx.scaleOf(sp)
                        idx.readRows(sp, from, cnt, buf)
                        for (k in 0 until cnt) {
                            var acc = 0f
                            val off = k * cfg.dim
                            for (d in 0 until cfg.dim) acc += qv[d] * buf[off + d]
                            // 2*s is the CSLS form; the correction term is subtracted below
                            scores[base + k] += 2.0 * w * acc * sc[from + k]
                        }
                    }
                    // CSLS hubness correction: subtract each row's mean similarity to the corpus
                    for (j in cfg.cslsSpaces.indices) {
                        val w = cfg.blend[cfg.cslsSpaces[j]] ?: continue
                        for (k in 0 until cnt) {
                            scores[base + k] -= w * idx.csls[(from + k) * 3 + j]
                        }
                    }
                }
            }
        }
        pool.invokeAll(tasks).forEach { it.get() }
        val t2 = System.currentTimeMillis()

        // ---- top-K ---------------------------------------------------------------------
        val k = minOf(cfg.topK, total)
        val top = topKIndices(scores, k)
        val topRows = IntArray(k) { rows[top[it]] }
        val sK = DoubleArray(k) { scores[top[it]] }

        val r = rerank(q, topRows, sK, k, total)
        val t3 = System.currentTimeMillis()
        return LocateResult(r.best.lat, r.best.lon, r.ranked,
            m, total, t1 - t0, t2 - t1, t3 - t2, r.confidence)
    }

    private class Reranked(
        val best: Candidate,
        val ranked: List<Candidate>,
        val confidence: Double,
    )

    /** E7: 16 features per candidate, a logistic score, blended back onto the retrieval score. */
    private fun rerank(
        q: Query, rows: IntArray, sK: DoubleArray, k: Int, nCandidates: Int,
    ): Reranked {
        val lat = DoubleArray(k) { idx.latOf(rows[it]) }
        val lon = DoubleArray(k) { idx.lonOf(rows[it]) }

        // raw-CLS similarity to the query, and candidate-to-candidate CLS cosines
        val clsScale = idx.scaleOf("cls")
        val buf = ByteArray(k * cfg.dim)
        for (i in 0 until k) idx.readRow("cls", rows[i], buf, i * cfg.dim)
        val bb = DoubleArray(k)
        val qc = q.spaces["cls"] ?: q.cls
        for (i in 0 until k) {
            var acc = 0f
            val off = i * cfg.dim
            for (d in 0 until cfg.dim) acc += qc[d] * buf[off + d]
            bb[i] = (acc * clsScale[rows[i]]).toDouble()
        }
        val unit = Array(k) { i ->
            val v = FloatArray(cfg.dim)
            var n = 0f
            val off = i * cfg.dim
            for (d in 0 until cfg.dim) { val x = buf[off + d].toFloat(); v[d] = x; n += x * x }
            val inv = 1f / (sqrt(n.toDouble()).toFloat() + 1e-9f)
            for (d in 0 until cfg.dim) v[d] *= inv
            v
        }
        val cc = Array(k) { i ->
            DoubleArray(k) { j ->
                var acc = 0f
                for (d in 0 until cfg.dim) acc += unit[i][d] * unit[j][d]
                acc.toDouble()
            }
        }
        val dkk = Array(k) { i -> DoubleArray(k) { j -> haversineKm(lat[i], lon[i], lat[j], lon[j]) } }

        // softmax over retrieval scores drives every consensus feature
        var mx = Double.NEGATIVE_INFINITY
        for (s in sK) if (s > mx) mx = s
        val w = DoubleArray(k) { exp((sK[it] - mx) / 0.05) }
        val wsum = w.sum() + 1e-12
        for (i in 0 until k) w[i] /= wsum

        val diffu = DoubleArray(k)
        for (i in 0 until k) {
            var rowSum = 0.0
            for (j in 0 until k) if (cc[i][j] > 0.5) rowSum += cc[i][j]
            rowSum += 1e-12
            var acc = 0.0
            for (j in 0 until k) if (cc[i][j] > 0.5) acc += (cc[i][j] / rowSum) * w[j]
            diffu[i] = acc
        }
        val cons10 = DoubleArray(k); val cons25 = DoubleArray(k); val cons50 = DoubleArray(k)
        val ccmean = DoubleArray(k); val ccmax3 = DoubleArray(k)
        for (i in 0 until k) {
            var a = 0.0; var b = 0.0; var c = 0.0; var m0 = 0.0
            for (j in 0 until k) {
                if (dkk[i][j] <= 10.0) a += w[j]
                if (dkk[i][j] <= 25.0) b += w[j]
                if (dkk[i][j] <= 50.0) c += w[j]
                m0 += cc[i][j] * w[j]
            }
            cons10[i] = a; cons25[i] = b; cons50[i] = c; ccmean[i] = m0
            val sorted = cc[i].clone(); sorted.sort()
            // mean of ranks -4..-2, i.e. the three highest excluding the self-similarity at -1
            ccmax3[i] = (sorted[k - 4] + sorted[k - 3] + sorted[k - 2]) / 3.0
        }
        var ent = 0.0
        for (i in 0 until k) ent -= w[i] * ln(w[i].coerceAtLeast(1e-12))
        val nvalid = k.toDouble() / cfg.topK

        val e7 = DoubleArray(k)
        val f = DoubleArray(16)
        for (i in 0 until k) {
            val margin = sK[0] - sK[i]
            f[0] = sK[i]; f[1] = bb[i]; f[2] = ln(1.0 + i); f[3] = margin
            f[4] = cons10[i]; f[5] = cons25[i]; f[6] = cons50[i]; f[7] = diffu[i] * 10
            f[8] = ccmean[i]; f[9] = ccmax3[i]; f[10] = ent; f[11] = nvalid
            f[12] = sK[i] * cons25[i]; f[13] = bb[i] * cons25[i]
            f[14] = margin * cons25[i]; f[15] = diffu[i] * cons25[i] * 10
            var acc = cfg.e7B
            for (d in 0 until 16) acc += cfg.e7W[d] * ((f[d] - cfg.e7Mu[d]) / cfg.e7Sd[d])
            e7[i] = acc
        }

        val out = ArrayList<Candidate>(k)
        var bestI = 0
        var bestV = Double.NEGATIVE_INFINITY
        var secondV = Double.NEGATIVE_INFINITY
        for (i in 0 until k) {
            val v = sK[i] / cfg.scoreSd + cfg.e7Lam * e7[i]
            out.add(Candidate(rows[i], lat[i], lon[i], sK[i], v))
            if (v > bestV) { secondV = bestV; bestV = v; bestI = i }
            else if (v > secondV) secondV = v
        }
        out.sortByDescending { it.rerank }

        // How much should the user trust this? Fitted offline against held-out ground truth
        // from signals already on hand, so it costs nothing beyond the arithmetic below.
        var a10 = 0.0; var a25 = 0.0; var a100 = 0.0
        for (j in 0 until k) {
            val d = dkk[bestI][j]
            if (d <= 10.0) a10 += w[j]
            if (d <= 25.0) a25 += w[j]
            if (d <= 100.0) a100 += w[j]
        }
        val feats = doubleArrayOf(
            a10, a25, a100, bb[bestI],
            if (k > 1) bestV - secondV else 0.0,
            ent, ln(1.0 + nCandidates)
        )
        var z = cfg.confB
        for (d in feats.indices) z += cfg.confW[d] * ((feats[d] - cfg.confMu[d]) / cfg.confSd[d])
        val confidence = 1.0 / (1.0 + exp(-z))

        return Reranked(out.first(), out, confidence)
    }

    override fun close() { pool.shutdownNow() }
}

/** Indices of the [k] largest values in [v], highest first. */
fun topKIndices(v: DoubleArray, k: Int): IntArray {
    val idx = IntArray(v.size) { it }
    // partial selection: cheaper than a full sort when the pool is tens of thousands of rows
    var lo = 0
    var hi = v.size - 1
    while (lo < hi) {
        val pivot = v[idx[(lo + hi) ushr 1]]
        var i = lo
        var j = hi
        while (i <= j) {
            while (v[idx[i]] > pivot) i++
            while (v[idx[j]] < pivot) j--
            if (i <= j) { val t = idx[i]; idx[i] = idx[j]; idx[j] = t; i++; j-- }
        }
        if (k - 1 <= j) hi = j else if (k - 1 >= i) lo = i else break
    }
    val out = idx.copyOfRange(0, k)
    val boxed = out.toTypedArray()
    java.util.Arrays.sort(boxed) { a, b -> v[b].compareTo(v[a]) }
    return IntArray(k) { boxed[it] }
}
