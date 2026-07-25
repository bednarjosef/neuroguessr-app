package com.neuroguessr.app

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.exponentialDecay
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.calculateCentroid
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.input.pointer.PointerEventPass
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChanged
import androidx.compose.ui.input.pointer.util.VelocityTracker
import androidx.compose.ui.platform.LocalHapticFeedback
import kotlinx.coroutines.launch
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

/**
 * Camera for an equirectangular world. `zoomLog` is log2 of "how many times the world width
 * fits the viewport", so pinch feels linear and a fling decays evenly.
 */
class MapCamera {
    val lon = Animatable(0f)
    val lat = Animatable(0f)
    val zoomLog = Animatable(0f)

    val zoom: Float get() = 2f.pow(zoomLog.value)
    fun pxPerDeg(widthPx: Float): Float = widthPx * zoom / 360f

    suspend fun flyTo(targetLon: Float, targetLat: Float, targetZoomLog: Float, ms: Int = 900) {
        val spec = tween<Float>(ms, easing = FastOutSlowInEasing)
        var t = targetLon
        while (t - lon.value > 180f) t -= 360f
        while (t - lon.value < -180f) t += 360f
        kotlinx.coroutines.coroutineScope {
            launch { lon.animateTo(t, spec) }
            launch { lat.animateTo(targetLat, spec) }
            launch { zoomLog.animateTo(targetZoomLog, spec) }
        }
        lon.snapTo(wrapLon(lon.value))
    }
}

private fun wrapLon(v: Float): Float {
    var x = v
    while (x > 180f) x -= 360f
    while (x < -180f) x += 360f
    return x
}

const val ZOOM_MIN = -0.25f
const val ZOOM_MAX = 9.0f

/**
 * Basemap and markers are drawn on SEPARATE canvases on purpose.
 *
 * The pin's pulse repeats forever; when it shared a canvas with the terrain it invalidated
 * every country polygon sixty times a second, so the map stuttered even while sitting still.
 * Split apart, the terrain redraws only when the camera actually moves, and the animated
 * layer costs a handful of circles.
 */
@Composable
fun MapCanvas(
    map: WorldMap,
    camera: MapCamera,
    pin: Pair<Double, Double>?,
    cluster: List<Candidate>,
    pinReveal: Float,
    modifier: Modifier = Modifier,
) {
    val scope = rememberCoroutineScope()
    val haptics = LocalHapticFeedback.current

    Box(
        modifier
            .pointerInput(Unit) {
                detectTapGestures(onDoubleTap = { pos ->
                    scope.launch {
                        haptics.performHapticFeedback(HapticFeedbackType.TextHandleMove)
                        val w = size.width.toFloat()
                        val ppd = camera.pxPerDeg(w)
                        val lonAt = camera.lon.value + (pos.x - w / 2f) / ppd
                        val latAt = camera.lat.value - (pos.y - size.height / 2f) / ppd
                        val target = min(ZOOM_MAX, camera.zoomLog.value + 1.5f)
                        val k = 1f - 2f.pow(camera.zoomLog.value - target)
                        camera.flyTo(
                            camera.lon.value + (lonAt - camera.lon.value) * k,
                            camera.lat.value + (latAt - camera.lat.value) * k,
                            target, 380
                        )
                    }
                })
            }
            .pointerInput(Unit) {
                val decay = exponentialDecay<Float>(frictionMultiplier = 1.1f)
                awaitEachGesture {
                    awaitFirstDown(requireUnconsumed = false)
                    val vt = VelocityTracker()
                    var pointers = 1
                    do {
                        val event = awaitPointerEvent(PointerEventPass.Main)
                        pointers = event.changes.count { it.pressed }
                        val zoomChange = event.calculateZoom()
                        val pan = event.calculatePan()
                        val centroid = event.calculateCentroid(useCurrent = true)
                        if (zoomChange != 1f || pan != Offset.Zero) {
                            val w = size.width.toFloat()
                            val ppdBefore = camera.pxPerDeg(w)
                            var newLon = camera.lon.value - pan.x / ppdBefore
                            var newLat = camera.lat.value + pan.y / ppdBefore
                            var newZoomLog = camera.zoomLog.value
                            if (zoomChange != 1f && centroid != Offset.Unspecified) {
                                newZoomLog = (camera.zoomLog.value +
                                        ln(zoomChange.toDouble()).toFloat() / ln(2f))
                                    .coerceIn(ZOOM_MIN, ZOOM_MAX)
                                val ppdAfter = w * 2f.pow(newZoomLog) / 360f
                                val cxOff = centroid.x - w / 2f
                                val cyOff = centroid.y - size.height / 2f
                                newLon = (newLon + cxOff / ppdBefore) - cxOff / ppdAfter
                                newLat = (newLat - cyOff / ppdBefore) + cyOff / ppdAfter
                            }
                            scope.launch {
                                camera.zoomLog.snapTo(newZoomLog)
                                camera.lon.snapTo(wrapLon(newLon))
                                camera.lat.snapTo(
                                    clampLat(newLat, newZoomLog, size.height.toFloat(), size.width.toFloat())
                                )
                            }
                            event.changes.forEach { if (it.positionChanged()) it.consume() }
                        }
                        val c = event.changes.firstOrNull { it.pressed }
                        if (c != null && pointers == 1) vt.addPosition(c.uptimeMillis, c.position)
                    } while (pointers > 0)

                    val v = vt.calculateVelocity()
                    val ppd = camera.pxPerDeg(size.width.toFloat())
                    if (abs(v.x) > 80f || abs(v.y) > 80f) {
                        scope.launch {
                            launch {
                                camera.lon.animateDecay(-v.x / ppd, decay)
                                camera.lon.snapTo(wrapLon(camera.lon.value))
                            }
                            launch { camera.lat.animateDecay(v.y / ppd, decay) }
                        }
                    }
                }
            }
    ) {
        BaseMap(map, camera, Modifier.fillMaxSize())
        MarkerLayer(camera, pin, cluster, pinReveal, Modifier.fillMaxSize())
    }
}

/** Terrain. Redraws only when the camera changes. */
@Composable
private fun BaseMap(map: WorldMap, camera: MapCamera, modifier: Modifier) {
    Canvas(modifier) {
        val w = size.width
        val h = size.height
        val ppd = camera.pxPerDeg(w)
        val cLon = camera.lon.value
        val cLat = camera.lat.value

        drawRect(Ink.Base)
        val yTop = h / 2f - (90f - cLat) * ppd
        val yBot = h / 2f - (-90f - cLat) * ppd
        drawRect(
            Ink.Water, topLeft = Offset(0f, max(0f, yTop)),
            size = Size(w, (min(h, yBot) - max(0f, yTop)).coerceAtLeast(0f))
        )

        val spanLon = 360f / camera.zoom
        val spanLat = spanLon * h / max(1f, w)
        val minLon = cLon - spanLon / 2; val maxLon = cLon + spanLon / 2
        val minLat = cLat - spanLat / 2; val maxLat = cLat + spanLat / 2

        val canvas = drawContext.canvas.nativeCanvas
        val paint = android.graphics.Paint().apply { isAntiAlias = true }
        val mtx = android.graphics.Matrix().apply {
            setScale(ppd, ppd)
            postTranslate(w / 2f - cLon * ppd, h / 2f + cLat * ppd)
        }

        fun layer(name: String, fill: Int?, stroke: Int?, strokeW: Float) {
            val g = map.geomFor(name)
            if (g.paths.isEmpty()) return
            canvas.save()
            canvas.concat(mtx)
            val reps = if (spanLon >= 360f) -1..1 else
                (Math.floor(((minLon + 180f) / 360f).toDouble()).toInt()..
                        Math.floor(((maxLon + 180f) / 360f).toDouble()).toInt())
            for (rep in reps) {
                val shift = rep * 360f
                canvas.save()
                canvas.translate(shift, 0f)
                for (i in g.paths.indices) {
                    val b = i * 4
                    if (g.bbox[b + 2] + shift < minLon || g.bbox[b] + shift > maxLon ||
                        g.bbox[b + 3] < minLat || g.bbox[b + 1] > maxLat
                    ) continue
                    if (fill != null) {
                        paint.style = android.graphics.Paint.Style.FILL
                        paint.color = fill
                        canvas.drawPath(g.paths[i], paint)
                    }
                    if (stroke != null) {
                        paint.style = android.graphics.Paint.Style.STROKE
                        paint.color = stroke
                        paint.strokeWidth = strokeW / ppd
                        canvas.drawPath(g.paths[i], paint)
                    }
                }
                canvas.restore()
            }
            canvas.restore()
        }

        // The 1:50m outlines carry 94k vertices against 9k for 1:110m, and below a regional
        // zoom the difference is invisible while the cost is not — the detailed set only
        // earns its keep once few enough countries are on screen to cull most of it away.
        val detailed = camera.zoom > 7f
        layer(
            if (detailed) "countries_50m" else "countries_110m",
            (if (detailed) Ink.LandHi else Ink.Land).toArgbInt(), Ink.Border.toArgbInt(), 1.1f
        )
        if (detailed) layer("lakes_50m", Ink.Water.toArgbInt(), null, 0f)

        drawGraticule(minLon, maxLon, minLat, maxLat, cLon, cLat, ppd, w, h)
        drawPlaceLabels(map, cLon, cLat, ppd, w, h, minLon, maxLon, minLat, maxLat, camera.zoom)
    }
}

/** Pin, halo and candidate cluster. Cheap enough to animate every frame. */
@Composable
private fun MarkerLayer(
    camera: MapCamera,
    pin: Pair<Double, Double>?,
    cluster: List<Candidate>,
    pinReveal: Float,
    modifier: Modifier,
) {
    // the pulse only exists while there is something to pulse around
    val halo = if (pin != null) {
        val t = rememberInfiniteTransition(label = "pulse")
        t.animateFloat(
            0f, 1f, infiniteRepeatable(tween(2400, easing = LinearEasing)), label = "halo"
        ).value
    } else 0f

    Canvas(modifier) {
        if (pin == null) return@Canvas
        val w = size.width
        val h = size.height
        val ppd = camera.pxPerDeg(w)
        val cLon = camera.lon.value
        val cLat = camera.lat.value

        if (pinReveal > 0.05f) {
            for (c in cluster) {
                val p = project(c.lat, c.lon, cLon, cLat, ppd, w, h) ?: continue
                drawCircle(Ink.Signal.copy(alpha = 0.20f * pinReveal), radius = 4.5f, center = p)
                drawCircle(Ink.Coast.copy(alpha = 0.9f * pinReveal), radius = 2.2f, center = p)
            }
        }
        project(pin.first, pin.second, cLon, cLat, ppd, w, h)?.let { drawPin(it, pinReveal, halo) }
    }
}

private fun clampLat(lat: Float, zoomLog: Float, h: Float, w: Float): Float {
    val spanLat = (360f / 2f.pow(zoomLog)) * h / max(1f, w)
    val limit = max(0f, 90f - spanLat / 2f)
    return lat.coerceIn(-limit, limit)
}

private fun project(
    lat: Double, lon: Double, cLon: Float, cLat: Float, ppd: Float, w: Float, h: Float,
): Offset? {
    var dl = lon.toFloat() - cLon
    while (dl > 180f) dl -= 360f
    while (dl < -180f) dl += 360f
    val x = w / 2f + dl * ppd
    val y = h / 2f - (lat.toFloat() - cLat) * ppd
    if (x < -200f || x > w + 200f || y < -200f || y > h + 200f) return null
    return Offset(x, y)
}

private fun DrawScope.drawGraticule(
    minLon: Float, maxLon: Float, minLat: Float, maxLat: Float,
    cLon: Float, cLat: Float, ppd: Float, w: Float, h: Float,
) {
    val stepChoices = floatArrayOf(90f, 45f, 30f, 15f, 10f, 5f, 2f, 1f, 0.5f, 0.2f, 0.1f)
    val step = stepChoices.firstOrNull { it * ppd < 300f && it * ppd > 70f } ?: return
    val yTop = (h / 2f - (90f - cLat) * ppd).coerceIn(0f, h)
    val yBot = (h / 2f - (-90f - cLat) * ppd).coerceIn(0f, h)
    var lon = Math.floor((minLon / step).toDouble()).toFloat() * step
    while (lon <= maxLon) {
        val x = w / 2f + (lon - cLon) * ppd
        drawLine(Ink.Grid, Offset(x, yTop), Offset(x, yBot), strokeWidth = 1f)
        lon += step
    }
    var lat = Math.floor((max(minLat, -90f) / step).toDouble()).toFloat() * step
    while (lat <= min(maxLat, 90f)) {
        val y = h / 2f - (lat - cLat) * ppd
        drawLine(Ink.Grid, Offset(0f, y), Offset(w, y), strokeWidth = 1f)
        lat += step
    }
}

private fun DrawScope.drawPin(p: Offset, reveal: Float, halo: Float) {
    val r = 13f + 44f * halo
    drawCircle(
        Ink.Signal.copy(alpha = 0.30f * (1f - halo) * reveal), radius = r, center = p,
        style = Stroke(width = 2f)
    )
    val c = Offset(p.x, p.y - (1f - reveal) * 30f)
    drawCircle(Ink.Signal.copy(alpha = 0.18f * reveal), radius = 20f * reveal, center = c)
    drawCircle(Ink.Base, radius = 8f * reveal, center = c)
    drawCircle(Ink.Signal, radius = 5f * reveal, center = c)
}

private val labelPaint = android.graphics.Paint().apply {
    isAntiAlias = true
    typeface = android.graphics.Typeface.create("sans-serif", android.graphics.Typeface.NORMAL)
}
private val labelHalo = android.graphics.Paint().apply {
    isAntiAlias = true
    style = android.graphics.Paint.Style.STROKE
    typeface = android.graphics.Typeface.create("sans-serif", android.graphics.Typeface.NORMAL)
}
private val dotPaint = android.graphics.Paint().apply { isAntiAlias = true }

/** Text measurement is expensive and the label set barely changes between frames. */
private val widthCache = HashMap<Long, Float>(512)

private fun measureCached(p: android.graphics.Paint, s: String, sizeKey: Int): Float {
    val k = (sizeKey.toLong() shl 40) xor s.hashCode().toLong()
    return widthCache.getOrPut(k) { p.measureText(s) }
}

/**
 * Country names far out, cities as you come in.
 *
 * The basemap stores places sorted by importance with a rank index, so each zoom level draws a
 * prefix of the table rather than scanning seven thousand rows every frame. Overlapping labels
 * are dropped by a running box test — because the table is importance-ordered, the label that
 * survives a collision is always the more significant place.
 */
private fun DrawScope.drawPlaceLabels(
    map: WorldMap, cLon: Float, cLat: Float, ppd: Float, w: Float, h: Float,
    minLon: Float, maxLon: Float, minLat: Float, maxLat: Float, zoom: Float,
) {
    val canvas = drawContext.canvas.nativeCanvas
    val taken = ArrayList<android.graphics.RectF>(80)

    fun draw(
        places: WorldMap.Places, maxRank: Int, sizePx: Float, colour: Int,
        withDot: Boolean, tracking: Float = 0f, limit: Int = 120,
    ) {
        labelPaint.textSize = sizePx
        labelPaint.color = colour
        labelPaint.letterSpacing = tracking
        labelHalo.textSize = sizePx
        labelHalo.strokeWidth = sizePx * 0.16f
        labelHalo.color = Ink.Base.copy(alpha = 0.85f).toArgbInt()
        labelHalo.letterSpacing = tracking
        var drawn = 0
        val n = places.visibleCount(maxRank)
        for (i in 0 until n) {
            val lon = places.lon(i)
            val lat = places.lat(i)
            if (lon < minLon || lon > maxLon || lat < minLat || lat > maxLat) continue
            val p = project(lat, lon, cLon, cLat, ppd, w, h) ?: continue
            val name = places.name(i)
            if (name.isEmpty()) continue
            val tw = measureCached(labelPaint, name, sizePx.toInt())
            val dx = if (withDot) sizePx * 0.42f else -tw / 2f
            val box = android.graphics.RectF(
                p.x + dx - 4f, p.y - sizePx * 0.75f, p.x + dx + tw + 4f, p.y + sizePx * 0.4f
            )
            if (taken.any { android.graphics.RectF.intersects(it, box) }) continue
            taken.add(box)
            if (withDot) {
                dotPaint.color = colour
                canvas.drawCircle(p.x, p.y, if (places.isCapital(i)) sizePx * 0.16f else sizePx * 0.11f, dotPaint)
            }
            val ty = p.y + sizePx * 0.35f
            canvas.drawText(name, p.x + dx, ty, labelHalo)
            canvas.drawText(name, p.x + dx, ty, labelPaint)
            if (++drawn >= limit) return
        }
    }

    // far out, only the countries are legible; the cities arrive as the ground resolves
    map.countryLabels?.let { c ->
        val rank = when {
            zoom < 1.6f -> 3
            zoom < 3.5f -> 5
            zoom < 14f -> 7
            else -> -1          // too close for country labels to mean anything
        }
        if (rank >= 0) {
            draw(c, rank, 26f, Ink.TextDim.copy(alpha = 0.75f).toArgbInt(),
                withDot = false, tracking = 0.12f, limit = 40)
        }
    }
    map.places?.let { p ->
        val rank = when {
            zoom < 2f -> -1
            zoom < 4f -> 1
            zoom < 9f -> 2
            zoom < 20f -> 3
            zoom < 45f -> 4
            zoom < 90f -> 6
            zoom < 200f -> 7
            zoom < 500f -> 8
            else -> 10
        }
        if (rank >= 0) {
            draw(p, rank, 29f, Ink.Text.copy(alpha = 0.86f).toArgbInt(), withDot = true)
        }
    }
}

fun Color.toArgbInt(): Int = android.graphics.Color.argb(
    (alpha * 255).toInt(), (red * 255).toInt(), (green * 255).toInt(), (blue * 255).toInt()
)
