package com.neuroguessr.app

import androidx.compose.animation.core.AnimationState
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.animate
import androidx.compose.animation.core.animateDecay
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
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.input.pointer.PointerEventPass
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChanged
import androidx.compose.ui.input.pointer.util.VelocityTracker
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalHapticFeedback
import kotlinx.coroutines.launch
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

/**
 * Camera over a square Web Mercator world.
 *
 * `zoomLog` is log2 of "how many times the world width fits the viewport", so pinch feels
 * linear and a fling decays evenly. The centre is kept as a Mercator northing rather than a
 * latitude: panning is then linear in screen space, which is what stops the map sliding at a
 * different speed than the finger as you move away from the equator.
 *
 * The three values are plain state, not `Animatable`s. A gesture handler runs in a restricted
 * suspend scope that cannot touch an Animatable at all, so the old code had to post every
 * drag frame to another coroutine — a frame of queueing that the finger could feel. Here the
 * gesture writes the numbers itself and the animations are the ones that go through a
 * coroutine, which is the way round it should have been.
 */
class MapCamera {
    var lon by mutableFloatStateOf(0f)

    /** Mercator northing of the view centre, world units, +-180. */
    var my by mutableFloatStateOf(0f)
    var zoomLog by mutableFloatStateOf(0f)

    var viewW by mutableFloatStateOf(0f)
        private set
    var viewH by mutableFloatStateOf(0f)
        private set

    /**
     * The zoom at which the world exactly covers the viewport.
     *
     * Below it the terrain would sit in a letterbox of empty background, so it is a hard floor
     * rather than a suggestion — on a tall phone that means the widest view is a little under
     * half the globe, which is the same trade every map app makes.
     */
    var minZoomLog by mutableFloatStateOf(0f)
        private set

    private var job: kotlinx.coroutines.Job? = null

    val zoom: Float get() = 2f.pow(zoomLog)
    fun pxPerUnit(widthPx: Float): Float = widthPx * zoom / 360f
    val centerLat: Float get() = mercLatf(my)

    fun setViewport(w: Float, h: Float) {
        if (w <= 0f || h <= 0f || (w == viewW && h == viewH)) return
        viewW = w
        viewH = h
        minZoomLog = ln(max(1f, h / w).toDouble()).toFloat() / ln(2f)
        zoomLog = clampZoom(zoomLog)
        my = my.coerceIn(-northingLimit(zoomLog), northingLimit(zoomLog))
    }

    /** How far the centre may travel before a pole would leave a gap at the edge. */
    fun northingLimit(atZoomLog: Float): Float {
        val ppu = viewW * 2f.pow(atZoomLog) / 360f
        return if (ppu <= 0f) 180f else max(0f, 180f - (viewH / ppu) / 2f)
    }

    fun clampZoom(v: Float) = v.coerceIn(minZoomLog, ZOOM_MAX)

    /** Move the camera and keep it legal, in one place, for every caller. */
    fun setCenter(newLon: Float, newMy: Float, newZoomLog: Float) {
        zoomLog = clampZoom(newZoomLog)
        val limit = northingLimit(zoomLog)
        my = newMy.coerceIn(-limit, limit)
        lon = wrapLon(newLon)
    }

    fun stopAnimation() {
        job?.cancel()
        job = null
    }

    /** Start a camera animation, replacing whatever was running. */
    fun launchAnimation(scope: kotlinx.coroutines.CoroutineScope, block: suspend MapCamera.() -> Unit) {
        stopAnimation()
        job = scope.launch { block() }
    }

    suspend fun flyTo(targetLon: Float, targetLat: Float, targetZoomLog: Float, ms: Int = 900) {
        val z = clampZoom(targetZoomLog)
        var t = targetLon
        while (t - lon > 180f) t -= 360f
        while (t - lon < -180f) t += 360f
        // the destination's own limit, so the flight lands somewhere it is allowed to stay
        val endMy = mercYf(targetLat).coerceIn(-northingLimit(z), northingLimit(z))
        val lon0 = lon; val my0 = my; val z0 = zoomLog
        animate(0f, 1f, animationSpec = tween(ms, easing = FastOutSlowInEasing)) { f, _ ->
            lon = lon0 + (t - lon0) * f
            my = my0 + (endMy - my0) * f
            zoomLog = z0 + (z - z0) * f
        }
        lon = wrapLon(lon)
    }

    suspend fun zoomTo(targetZoomLog: Float, ms: Int = 700) {
        val z = clampZoom(targetZoomLog)
        val z0 = zoomLog
        val my0 = my
        animate(0f, 1f, animationSpec = tween(ms, easing = FastOutSlowInEasing)) { f, _ ->
            zoomLog = z0 + (z - z0) * f
            my = my0.coerceIn(-northingLimit(zoomLog), northingLimit(zoomLog))
        }
    }

    /**
     * Coast to a stop, easing into the pole limit rather than being clipped at it.
     *
     * The friction is the difference between a map that glides and one that stops dead under
     * your finger; lower than this and a hard flick keeps sliding for three seconds, which
     * reads as the map having got away from you.
     */
    suspend fun fling(vx: Float, vy: Float) {
        val decay = exponentialDecay<Float>(frictionMultiplier = 0.85f)
        kotlinx.coroutines.coroutineScope {
            launch {
                AnimationState(lon, vx).animateDecay(decay) { lon = value }
                lon = wrapLon(lon)
            }
            launch {
                val limit = northingLimit(zoomLog)
                AnimationState(my, vy).animateDecay(decay) {
                    val c = value.coerceIn(-limit, limit)
                    my = c
                    if (c != value) cancelAnimation()
                }
            }
        }
    }
}

private fun wrapLon(v: Float): Float {
    var x = v
    while (x > 180f) x -= 360f
    while (x < -180f) x += 360f
    return x
}

const val ZOOM_MAX = 11.0f

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
            .onSizeChanged { camera.setViewport(it.width.toFloat(), it.height.toFloat()) }
            .pointerInput(Unit) {
                detectTapGestures(onDoubleTap = { pos ->
                    haptics.performHapticFeedback(HapticFeedbackType.TextHandleMove)
                    val w = size.width.toFloat()
                    val ppu = camera.pxPerUnit(w)
                    val lonAt = camera.lon + (pos.x - w / 2f) / ppu
                    val myAt = camera.my - (pos.y - size.height / 2f) / ppu
                    val target = camera.clampZoom(camera.zoomLog + 1.5f)
                    // keep the tapped point under the finger as the scale changes
                    val k = 1f - 2f.pow(camera.zoomLog - target)
                    camera.launchAnimation(scope) {
                        flyTo(
                            camera.lon + (lonAt - camera.lon) * k,
                            mercLatf(camera.my + (myAt - camera.my) * k),
                            target, 380
                        )
                    }
                })
            }
            .pointerInput(Unit) {
                awaitEachGesture {
                    awaitFirstDown(requireUnconsumed = false)
                    camera.stopAnimation()
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
                            val ppuBefore = camera.pxPerUnit(w)
                            var newLon = camera.lon - pan.x / ppuBefore
                            var newMy = camera.my + pan.y / ppuBefore
                            var newZoomLog = camera.zoomLog
                            if (zoomChange != 1f && centroid != Offset.Unspecified) {
                                newZoomLog = camera.clampZoom(
                                    camera.zoomLog + ln(zoomChange.toDouble()).toFloat() / ln(2f)
                                )
                                val ppuAfter = w * 2f.pow(newZoomLog) / 360f
                                val cxOff = centroid.x - w / 2f
                                val cyOff = centroid.y - size.height / 2f
                                newLon = (newLon + cxOff / ppuBefore) - cxOff / ppuAfter
                                newMy = (newMy - cyOff / ppuBefore) + cyOff / ppuAfter
                            }
                            camera.setCenter(newLon, newMy, newZoomLog)
                            event.changes.forEach { if (it.positionChanged()) it.consume() }
                        }
                        val c = event.changes.firstOrNull { it.pressed }
                        if (c != null && pointers == 1) vt.addPosition(c.uptimeMillis, c.position)
                    } while (pointers > 0)

                    val v = vt.calculateVelocity()
                    val ppu = camera.pxPerUnit(size.width.toFloat())
                    if (abs(v.x) > 90f || abs(v.y) > 90f) {
                        camera.launchAnimation(scope) { fling(-v.x / ppu, v.y / ppu) }
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
        val ppu = camera.pxPerUnit(w)
        val cLon = camera.lon
        val cMy = camera.my

        drawRect(Ink.Base)
        val yTop = h / 2f - (180f - cMy) * ppu
        val yBot = h / 2f - (-180f - cMy) * ppu
        drawRect(
            Ink.Water, topLeft = Offset(0f, max(0f, yTop)),
            size = Size(w, (min(h, yBot) - max(0f, yTop)).coerceAtLeast(0f))
        )

        val spanLon = 360f / camera.zoom
        val spanY = h / ppu
        val minLon = cLon - spanLon / 2; val maxLon = cLon + spanLon / 2
        val minLat = mercLatf(cMy - spanY / 2); val maxLat = mercLatf(cMy + spanY / 2)

        val canvas = drawContext.canvas.nativeCanvas
        val paint = android.graphics.Paint().apply { isAntiAlias = true }
        val mtx = android.graphics.Matrix().apply {
            setScale(ppu, ppu)
            postTranslate(w / 2f - cLon * ppu, h / 2f + cMy * ppu)
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
                        paint.strokeWidth = strokeW / ppu
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
        //
        // Two passes per polygon is the whole budget. A third — a wide low-alpha shelf around
        // every coast — measured 2 ms a frame out of 16 for something you had to look for.
        val detailed = camera.zoom > 7f
        layer(
            if (detailed) "countries_50m" else "countries_110m",
            (if (detailed) Ink.LandHi else Ink.Land).toArgbInt(),
            Ink.Border.copy(alpha = 0.75f).toArgbInt(), 1.0f
        )
        if (detailed) layer("lakes_50m", Ink.Water.toArgbInt(), null, 0f)

        drawPlaceLabels(map, cLon, cMy, ppu, w, h, minLon, maxLon, minLat, maxLat, camera.zoom)
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
            0f, 1f, infiniteRepeatable(tween(2800, easing = LinearEasing)), label = "halo"
        ).value
    } else 0f

    Canvas(modifier) {
        if (pin == null) return@Canvas
        val w = size.width
        val h = size.height
        val ppu = camera.pxPerUnit(w)
        val cLon = camera.lon
        val cMy = camera.my

        if (pinReveal > 0.05f) {
            for (c in cluster) {
                val p = project(c.lat, c.lon, cLon, cMy, ppu, w, h) ?: continue
                drawCircle(Ink.Signal.copy(alpha = 0.16f * pinReveal), radius = 5.5f, center = p)
                drawCircle(Ink.Signal.copy(alpha = 0.55f * pinReveal), radius = 2.0f, center = p)
            }
        }
        project(pin.first, pin.second, cLon, cMy, ppu, w, h)?.let { drawPin(it, pinReveal, halo) }
    }
}

private fun project(
    lat: Double, lon: Double, cLon: Float, cMy: Float, ppu: Float, w: Float, h: Float,
): Offset? {
    var dl = lon.toFloat() - cLon
    while (dl > 180f) dl -= 360f
    while (dl < -180f) dl += 360f
    val x = w / 2f + dl * ppu
    val y = h / 2f - (mercY(lat).toFloat() - cMy) * ppu
    if (x < -200f || x > w + 200f || y < -200f || y > h + 200f) return null
    return Offset(x, y)
}

/**
 * A map pin, not a dot.
 *
 * The tip sits on the coordinate and the body stands above it, so the answer is never hidden
 * under its own marker — and a dark rim plus a ground shadow keep it readable over both the
 * pale land fill and the near-black ocean.
 */
private fun DrawScope.drawPin(p: Offset, reveal: Float, halo: Float) {
    if (reveal <= 0.01f) return
    val s = reveal
    val drop = (1f - reveal) * 46f
    val tip = Offset(p.x, p.y - drop)

    // the coordinate itself stays marked even while the pin is still falling
    drawCircle(Ink.Signal.copy(alpha = 0.85f * s), radius = 2.6f, center = p)
    val r = 14f + 46f * halo
    drawCircle(
        Ink.Signal.copy(alpha = 0.28f * (1f - halo) * s), radius = r, center = p,
        style = Stroke(width = 2f)
    )

    val head = Offset(tip.x, tip.y - 30f * s)
    val rad = 11.5f * s
    drawOval(
        Ink.Base.copy(alpha = 0.35f * s * (1f - (1f - reveal))),
        topLeft = Offset(p.x - 7f, p.y - 3f), size = Size(14f, 6f)
    )
    val body = Path().apply {
        moveTo(tip.x, tip.y)
        cubicTo(
            tip.x - rad * 0.62f, tip.y - rad * 1.35f,
            tip.x - rad, tip.y - rad * 1.9f,
            head.x - rad * 0.72f, head.y + rad * 0.7f
        )
        cubicTo(
            head.x - rad * 1.5f, head.y - rad * 0.9f,
            head.x + rad * 1.5f, head.y - rad * 0.9f,
            head.x + rad * 0.72f, head.y + rad * 0.7f
        )
        cubicTo(
            tip.x + rad, tip.y - rad * 1.9f,
            tip.x + rad * 0.62f, tip.y - rad * 1.35f,
            tip.x, tip.y
        )
        close()
    }
    drawPath(body, Ink.Base.copy(alpha = 0.55f), style = Stroke(width = 5f))
    drawPath(body, Ink.Signal)
    drawCircle(Color.White.copy(alpha = 0.92f), radius = rad * 0.36f, center = head)
}

private val labelPaint = android.graphics.Paint().apply {
    isAntiAlias = true
    typeface = android.graphics.Typeface.create("sans-serif-medium", android.graphics.Typeface.NORMAL)
}
private val labelHalo = android.graphics.Paint().apply {
    isAntiAlias = true
    style = android.graphics.Paint.Style.STROKE
    strokeJoin = android.graphics.Paint.Join.ROUND
    typeface = android.graphics.Typeface.create("sans-serif-medium", android.graphics.Typeface.NORMAL)
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
    map: WorldMap, cLon: Float, cMy: Float, ppu: Float, w: Float, h: Float,
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
        labelHalo.strokeWidth = sizePx * 0.20f
        labelHalo.color = Ink.Base.copy(alpha = 0.72f).toArgbInt()
        labelHalo.letterSpacing = tracking
        var drawn = 0
        val n = places.visibleCount(maxRank)
        for (i in 0 until n) {
            val lon = places.lon(i)
            val lat = places.lat(i)
            if (lon < minLon || lon > maxLon || lat < minLat || lat > maxLat) continue
            val p = project(lat, lon, cLon, cMy, ppu, w, h) ?: continue
            val name = places.name(i)
            if (name.isEmpty()) continue
            val tw = measureCached(labelPaint, name, sizePx.toInt())
            val dx = if (withDot) sizePx * 0.46f else -tw / 2f
            val box = android.graphics.RectF(
                p.x + dx - 5f, p.y - sizePx * 0.78f, p.x + dx + tw + 5f, p.y + sizePx * 0.42f
            )
            if (taken.any { android.graphics.RectF.intersects(it, box) }) continue
            taken.add(box)
            if (withDot) {
                dotPaint.color = Ink.Base.copy(alpha = 0.7f).toArgbInt()
                canvas.drawCircle(p.x, p.y, sizePx * (if (places.isCapital(i)) 0.22f else 0.17f), dotPaint)
                dotPaint.color = colour
                canvas.drawCircle(p.x, p.y, sizePx * (if (places.isCapital(i)) 0.15f else 0.10f), dotPaint)
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
            zoom < 2.4f -> 3
            zoom < 4.5f -> 5
            zoom < 14f -> 7
            else -> -1          // too close for country labels to mean anything
        }
        if (rank >= 0) {
            draw(c, rank, 27f, Ink.Coast.copy(alpha = 0.95f).toArgbInt(),
                withDot = false, tracking = 0.14f, limit = 40)
        }
    }
    map.places?.let { p ->
        val rank = when {
            zoom < 2.6f -> -1
            zoom < 5f -> 1
            zoom < 10f -> 2
            zoom < 22f -> 3
            zoom < 50f -> 4
            zoom < 100f -> 6
            zoom < 220f -> 7
            zoom < 520f -> 8
            else -> 10
        }
        if (rank >= 0) {
            draw(p, rank, 29f, Ink.Text.copy(alpha = 0.90f).toArgbInt(), withDot = true)
        }
    }
}

fun Color.toArgbInt(): Int = android.graphics.Color.argb(
    (alpha * 255).toInt(), (red * 255).toInt(), (green * 255).toInt(), (blue * 255).toInt()
)
