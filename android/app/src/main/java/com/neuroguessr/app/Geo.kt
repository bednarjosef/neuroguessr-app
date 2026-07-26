package com.neuroguessr.app

import kotlin.math.abs
import kotlin.math.asin
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sin
import kotlin.math.sqrt

const val EARTH_KM = 6371.0088

fun haversineKm(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
    val p = Math.PI / 180.0
    val a1 = lat1 * p; val o1 = lon1 * p; val a2 = lat2 * p; val o2 = lon2 * p
    val h = sin((a2 - a1) / 2).let { it * it } +
            cos(a1) * cos(a2) * sin((o2 - o1) / 2).let { it * it }
    return 2 * EARTH_KM * asin(sqrt(h.coerceIn(0.0, 1.0)))
}

/** GeoGuessr round score for a guess this far from the truth. */
fun geoguessrScore(km: Double): Double = 5000.0 * exp(-km / 1492.7)

/** Softmax in place over [v], returning the same array. */
fun softmaxInPlace(v: FloatArray): FloatArray {
    var mx = Float.NEGATIVE_INFINITY
    for (x in v) if (x > mx) mx = x
    var s = 0.0
    for (i in v.indices) { val e = exp((v[i] - mx).toDouble()); v[i] = e.toFloat(); s += e }
    val inv = (1.0 / s).toFloat()
    for (i in v.indices) v[i] *= inv
    return v
}

fun logClamped(x: Float): Double = ln(x.toDouble().coerceAtLeast(1e-12))

/** Human-readable distance, matching how people talk about map error. */
fun formatKm(km: Double): String = when {
    km < 1.0 -> "${(km * 1000).toInt()} m"
    km < 10.0 -> String.format(java.util.Locale.ROOT, "%.2f km", km)
    km < 1000.0 -> String.format(java.util.Locale.ROOT, "%.1f km", km)
    else -> String.format(java.util.Locale.ROOT, "%,.0f km", km)
}

/**
 * Web Mercator, in "world units" chosen so the map is a 360x360 square.
 *
 * Equirectangular gives a 2:1 world, which can never fill a portrait viewport without the
 * camera sliding off the poles — that is what left dead space above and below the map. In
 * Mercator the world is square, latitude is cut at +-85.051 deg, and northing lands in the
 * same +-180 range as longitude, so one pixels-per-unit scale drives both axes and the
 * pre-built geometry still transforms with a plain matrix.
 */
const val MERC_LAT_MAX = 85.05112878

fun mercY(latDeg: Double): Double {
    val lat = latDeg.coerceIn(-MERC_LAT_MAX, MERC_LAT_MAX) * (Math.PI / 180.0)
    return ln(kotlin.math.tan(Math.PI / 4.0 + lat / 2.0)) * (180.0 / Math.PI)
}

fun mercLat(y: Double): Double =
    (2.0 * kotlin.math.atan(exp(y * (Math.PI / 180.0))) - Math.PI / 2.0) * (180.0 / Math.PI)

fun mercYf(latDeg: Float): Float = mercY(latDeg.toDouble()).toFloat()
fun mercLatf(y: Float): Float = mercLat(y.toDouble()).toFloat()

fun formatLatLon(lat: Double, lon: Double): String {
    val ns = if (lat >= 0) "N" else "S"
    val ew = if (lon >= 0) "E" else "W"
    return String.format(java.util.Locale.ROOT, "%.5f°%s, %.5f°%s", abs(lat), ns, abs(lon), ew)
}
