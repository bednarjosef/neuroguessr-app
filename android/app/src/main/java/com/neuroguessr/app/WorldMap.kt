package com.neuroguessr.app

import android.content.Context
import android.graphics.Path
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Reader for the offline basemap (`world.bin`, magic NGWM v1/v2) — Natural Earth geometry
 * quantised to int32 microdegrees. No network, no tiles, no map SDK.
 *
 * Layout: header -> layer table -> feature table -> ring table -> coordinate blob -> names.
 * Polygon rings omit the closing vertex, and ring 0 of a feature is its exterior; the rest
 * are holes. All offsets are read from the header rather than assumed.
 */
class WorldMap(private val buf: ByteBuffer) {

    class Layer(val nameId: Int, val firstFeature: Int, val featureCount: Int, val geomType: Int)

    private val featureOff: Int
    private val ringOff: Int
    private val coordOff: Int
    private val nameTableOff: Int
    private val nameBlobOff: Int
    val coordScale: Double
    val featureCount: Int
    val layers: List<Layer>
    private val nameCount: Int
    private val version: Int
    private var placeOff = 0
    private var placeStride = 24
    private var rankIndexOff = 0
    private var rankIndexEntries = 16

    init {
        require(buf.getInt(0) == 0x4D57474E) { "not a NGWM basemap" }
        // v2 added a 128-byte header and POINT layers; the polygon/line sections and every
        // offset this class reads are unchanged, so v1 and v2 both decode with this code.
        version = buf.getInt(4)
        require(version in 1..2) { "unsupported basemap version $version" }
        val layerCount = buf.getInt(8)
        featureCount = buf.getInt(12)
        nameCount = buf.getInt(24)
        coordScale = buf.getInt(28).toDouble()
        val layerOff = buf.getInt(32)
        featureOff = buf.getInt(36)
        ringOff = buf.getInt(40)
        coordOff = buf.getInt(44)
        nameTableOff = buf.getInt(48)
        nameBlobOff = buf.getInt(52)
        layers = (0 until layerCount).map { i ->
            val o = layerOff + i * 16
            Layer(buf.getInt(o), buf.getInt(o + 4), buf.getInt(o + 8), buf.getInt(o + 12))
        }
        if (version >= 2) {
            placeOff = buf.getInt(72)
            placeStride = buf.getInt(76)
            rankIndexOff = buf.getInt(80)
            rankIndexEntries = buf.getInt(84)
        }
    }

    fun layer(name: String): Layer? = layers.firstOrNull { nameOf(it.nameId) == name }

    /**
     * Populated places (format v2+). Null on older basemaps, so the renderer degrades to an
     * unlabelled map rather than refusing to start.
     *
     * Records are pre-sorted by importance and the file carries a rank index, so drawing "every
     * place worth showing at this zoom" is a prefix of the table — no per-frame filtering of
     * seven thousand rows.
     */
    inner class Places(
        private val first: Int,
        override_count: Int,
        private val rankBlock: Int,
    ) {
        val count = override_count
        private fun rec(i: Int) = placeOff + (first + i) * placeStride
        fun lon(i: Int) = buf.getInt(rec(i) + 4) / coordScale
        fun lat(i: Int) = buf.getInt(rec(i) + 8) / coordScale
        fun rank(i: Int) = buf.get(rec(i) + 20).toInt() and 0xFF
        fun flags(i: Int) = buf.get(rec(i) + 21).toInt() and 0xFF
        fun isCapital(i: Int) = (flags(i) and 0x01) != 0
        fun name(i: Int) = nameOf(buf.getInt(rec(i)))

        /** How many leading records have rank <= [maxRank]. */
        fun visibleCount(maxRank: Int): Int {
            if (rankBlock < 0 || maxRank >= rankIndexEntries - 1) return count
            val o = rankIndexOff + (rankBlock * rankIndexEntries + maxRank + 1) * 4
            return buf.getInt(o).coerceIn(0, count)
        }
    }

    /** Cities and towns, most important first. */
    val places: Places? by lazy { pointLayer("places_10m") }

    /** One label anchor per country, guaranteed to sit inside its own borders. */
    val countryLabels: Places? by lazy { pointLayer("country_labels") }

    private fun pointLayer(name: String): Places? {
        if (version < 2) return null
        val l = layers.firstOrNull { nameOf(it.nameId) == name && it.geomType == 2 } ?: return null
        val block = layers.filter { it.geomType == 2 }.indexOfFirst { it.nameId == l.nameId }
        return Places(l.firstFeature, l.featureCount, block)
    }

    fun nameOf(id: Int): String {
        if (id <= 0 || id >= nameCount) return ""
        val a = buf.getInt(nameTableOff + id * 4)
        val b = buf.getInt(nameTableOff + (id + 1) * 4)
        if (b <= a) return ""
        val bytes = ByteArray(b - a)
        val dup = buf.duplicate().order(ByteOrder.LITTLE_ENDIAN)
        dup.position(nameBlobOff + a)
        dup.get(bytes)
        return String(bytes, Charsets.UTF_8)
    }

    fun featureNameId(f: Int) = buf.getInt(featureOff + f * 28)
    fun featureFirstRing(f: Int) = buf.getInt(featureOff + f * 28 + 4)
    fun featureRingCount(f: Int) = buf.getInt(featureOff + f * 28 + 8)
    fun featureMinLon(f: Int) = buf.getFloat(featureOff + f * 28 + 12)
    fun featureMinLat(f: Int) = buf.getFloat(featureOff + f * 28 + 16)
    fun featureMaxLon(f: Int) = buf.getFloat(featureOff + f * 28 + 20)
    fun featureMaxLat(f: Int) = buf.getFloat(featureOff + f * 28 + 24)
    private fun ringFirstPoint(r: Int) = buf.getInt(ringOff + r * 8)
    private fun ringPointCount(r: Int) = buf.getInt(ringOff + r * 8 + 4)
    fun lonAt(p: Int) = buf.getInt(coordOff + p * 8) / coordScale
    fun latAt(p: Int) = buf.getInt(coordOff + p * 8 + 4) / coordScale

    /**
     * Build android Paths in "world units": x = lon, y = -lat, so a caller can scale/translate
     * without rebuilding geometry. Cached per layer because this walks every vertex.
     */
    /** Paths in world units plus a flat bbox table, so a frame can cull before it draws. */
    class Geom(val paths: List<Path>, val bbox: FloatArray)

    private val geomCache = HashMap<String, Geom>()

    fun geomFor(layerName: String): Geom = geomCache.getOrPut(layerName) {
        val l = layer(layerName) ?: return@getOrPut Geom(emptyList(), FloatArray(0))
        val out = ArrayList<Path>(l.featureCount)
        val bb = FloatArray(l.featureCount * 4)
        for ((i, f) in (l.firstFeature until l.firstFeature + l.featureCount).withIndex()) {
            val p = Path()
            val fr = featureFirstRing(f)
            for (r in fr until fr + featureRingCount(f)) {
                val fp = ringFirstPoint(r)
                val n = ringPointCount(r)
                if (n < 2) continue
                p.moveTo(lonAt(fp).toFloat(), (-latAt(fp)).toFloat())
                for (k in 1 until n) p.lineTo(lonAt(fp + k).toFloat(), (-latAt(fp + k)).toFloat())
                if (l.geomType == 0) p.close()
            }
            out.add(p)
            bb[i * 4] = featureMinLon(f); bb[i * 4 + 1] = featureMinLat(f)
            bb[i * 4 + 2] = featureMaxLon(f); bb[i * 4 + 3] = featureMaxLat(f)
        }
        Geom(out, bb)
    }

    fun pathsFor(layerName: String): List<Path> = geomFor(layerName).paths

    // ---- country lookup ---------------------------------------------------------------

    private fun ringContains(r: Int, lat: Double, lon: Double): Boolean {
        val fp = ringFirstPoint(r)
        val n = ringPointCount(r)
        var inside = false
        var j = n - 1
        for (i in 0 until n) {
            val xi = lonAt(fp + i); val yi = latAt(fp + i)
            val xj = lonAt(fp + j); val yj = latAt(fp + j)
            if ((yi > lat) != (yj > lat) &&
                lon < (xj - xi) * (lat - yi) / (yj - yi) + xi
            ) inside = !inside
            j = i
        }
        return inside
    }

    /** Distance in km from a point to a ring's nearest edge, equirectangular approximation. */
    private fun ringDistanceKm(r: Int, lat: Double, lon: Double): Double {
        val fp = ringFirstPoint(r)
        val n = ringPointCount(r)
        val kx = 111.32 * cos(Math.toRadians(lat))
        val ky = 110.57
        var best = Double.MAX_VALUE
        var j = n - 1
        for (i in 0 until n) {
            val ax = (lonAt(fp + j) - lon) * kx; val ay = (latAt(fp + j) - lat) * ky
            val bx = (lonAt(fp + i) - lon) * kx; val by = (latAt(fp + i) - lat) * ky
            val dx = bx - ax; val dy = by - ay
            val len2 = dx * dx + dy * dy
            val t = if (len2 <= 0.0) 0.0 else (((-ax) * dx + (-ay) * dy) / len2).coerceIn(0.0, 1.0)
            val px = ax + t * dx; val py = ay + t * dy
            val d = sqrt(px * px + py * py)
            if (d < best) best = d
            j = i
        }
        return best
    }

    /**
     * Name the country a coordinate falls in. Predictions routinely land just offshore —
     * Manhattan is not even land at 1:50m — so a strict miss falls back to the nearest
     * coastline within [snapKm], and only then gives up.
     */
    fun countryAt(lat: Double, lon: Double, snapKm: Double = 50.0): String? {
        val l = layer("countries_50m") ?: return null
        var nearest: String? = null
        var nearestKm = snapKm
        for (f in l.firstFeature until l.firstFeature + l.featureCount) {
            if (lon < featureMinLon(f) || lon > featureMaxLon(f) ||
                lat < featureMinLat(f) || lat > featureMaxLat(f)
            ) continue
            val fr = featureFirstRing(f)
            if (ringContains(fr, lat, lon)) {
                var inHole = false
                for (r in fr + 1 until fr + featureRingCount(f)) {
                    if (ringContains(r, lat, lon)) { inHole = true; break }
                }
                if (!inHole) return nameOf(featureNameId(f))
            }
        }
        // strict miss: snap to the nearest coast, scanning only features whose bbox is close
        val pad = snapKm / 111.0
        for (f in l.firstFeature until l.firstFeature + l.featureCount) {
            if (lon < featureMinLon(f) - pad || lon > featureMaxLon(f) + pad ||
                lat < featureMinLat(f) - pad || lat > featureMaxLat(f) + pad
            ) continue
            val d = ringDistanceKm(featureFirstRing(f), lat, lon)
            if (d < nearestKm) { nearestKm = d; nearest = nameOf(featureNameId(f)) }
        }
        return nearest
    }

    companion object {
        fun fromAssets(ctx: Context, name: String = "world.bin"): WorldMap {
            val bytes = ctx.assets.open(name).use { it.readBytes() }
            return WorldMap(ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN))
        }
    }
}

/** Equirectangular viewport: world units are (lon, -lat). */
class MapView2D(var centerLon: Double = 0.0, var centerLat: Double = 20.0, var zoom: Double = 1.0) {
    fun clamp(widthPx: Float, heightPx: Float) {
        zoom = zoom.coerceIn(0.6, 400.0)
        val spanLon = 360.0 / zoom
        val spanLat = spanLon * heightPx / max(1f, widthPx)
        centerLat = centerLat.coerceIn(-90.0 + min(90.0, spanLat / 2), 90.0 - min(90.0, spanLat / 2))
        centerLon = ((centerLon + 180.0).mod(360.0)) - 180.0
        if (abs(spanLon) >= 360.0) centerLon = 0.0
    }
}
