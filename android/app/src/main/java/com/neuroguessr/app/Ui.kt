package com.neuroguessr.app

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.ripple
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

// ---- small pieces ---------------------------------------------------------------------

@Composable
fun Hairline(modifier: Modifier = Modifier, color: Color = Ink.HairSoft) =
    Box(modifier.fillMaxWidth().height(1.dp).background(color))

/** Quiet supporting text. Sentence case: the interface is talking, not labelling a diagram. */
@Composable
fun Caption(text: String, color: Color = Ink.TextDim, modifier: Modifier = Modifier) {
    Text(text, color = color, style = MaterialTheme.typography.bodySmall, modifier = modifier)
}

/**
 * Icons drawn rather than imported.
 *
 * The handful this app needs — a photo, a camera, a clipboard, a crosshair — are a few strokes
 * each, and drawing them keeps every glyph on the same 1.7dp stem as the rest of the interface
 * instead of importing a multi-megabyte icon font for four shapes.
 */
enum class Glyph { PHOTO, CAMERA, COPY, TARGET, PIN, DOWNLOAD }

@Composable
fun Icon(glyph: Glyph, tint: Color, size: Int = 20, modifier: Modifier = Modifier) {
    Canvas(modifier.size(size.dp)) { drawGlyph(glyph, tint) }
}

private fun DrawScope.drawGlyph(glyph: Glyph, tint: Color) {
    val s = size.minDimension
    val sw = s * 0.085f
    val st = Stroke(width = sw, cap = StrokeCap.Round)
    fun rr(l: Float, t: Float, r: Float, b: Float, rad: Float) =
        androidx.compose.ui.geometry.RoundRect(
            l * s, t * s, r * s, b * s,
            androidx.compose.ui.geometry.CornerRadius(rad * s, rad * s)
        )
    when (glyph) {
        Glyph.PHOTO -> {
            drawPath(Path().apply { addRoundRect(rr(0.10f, 0.16f, 0.90f, 0.84f, 0.14f)) }, tint, style = st)
            drawCircle(tint, radius = s * 0.075f, center = Offset(s * 0.34f, s * 0.36f))
            drawPath(
                Path().apply {
                    moveTo(s * 0.14f, s * 0.74f); lineTo(s * 0.40f, s * 0.48f)
                    lineTo(s * 0.62f, s * 0.70f); lineTo(s * 0.74f, s * 0.58f)
                    lineTo(s * 0.87f, s * 0.72f)
                }, tint, style = st
            )
        }
        Glyph.CAMERA -> {
            drawPath(Path().apply { addRoundRect(rr(0.07f, 0.27f, 0.93f, 0.85f, 0.16f)) }, tint, style = st)
            drawPath(
                Path().apply {
                    moveTo(s * 0.34f, s * 0.27f); lineTo(s * 0.41f, s * 0.14f)
                    lineTo(s * 0.59f, s * 0.14f); lineTo(s * 0.66f, s * 0.27f)
                }, tint, style = st
            )
            drawCircle(tint, radius = s * 0.155f, center = Offset(s * 0.5f, s * 0.58f), style = st)
        }
        Glyph.COPY -> {
            drawPath(Path().apply { addRoundRect(rr(0.30f, 0.10f, 0.90f, 0.66f, 0.13f)) }, tint, style = st)
            drawPath(
                Path().apply {
                    moveTo(s * 0.70f, s * 0.78f)
                    cubicTo(s * 0.70f, s * 0.86f, s * 0.66f, s * 0.90f, s * 0.58f, s * 0.90f)
                    lineTo(s * 0.22f, s * 0.90f)
                    cubicTo(s * 0.14f, s * 0.90f, s * 0.10f, s * 0.86f, s * 0.10f, s * 0.78f)
                    lineTo(s * 0.10f, s * 0.42f)
                    cubicTo(s * 0.10f, s * 0.34f, s * 0.14f, s * 0.30f, s * 0.22f, s * 0.30f)
                }, tint, style = st
            )
        }
        Glyph.TARGET -> {
            drawCircle(tint, radius = s * 0.28f, center = Offset(s / 2, s / 2), style = st)
            drawCircle(tint, radius = s * 0.065f, center = Offset(s / 2, s / 2))
            for (a in 0 until 4) {
                val dx = if (a % 2 == 0) 1f else 0f
                val dy = 1f - dx
                val sign = if (a < 2) -1f else 1f
                drawLine(
                    tint,
                    Offset(s / 2 + dx * sign * s * 0.34f, s / 2 + dy * sign * s * 0.34f),
                    Offset(s / 2 + dx * sign * s * 0.46f, s / 2 + dy * sign * s * 0.46f),
                    strokeWidth = sw, cap = StrokeCap.Round
                )
            }
        }
        Glyph.DOWNLOAD -> {
            drawLine(
                tint, Offset(s * 0.5f, s * 0.10f), Offset(s * 0.5f, s * 0.56f),
                strokeWidth = sw, cap = StrokeCap.Round
            )
            drawPath(
                Path().apply {
                    moveTo(s * 0.30f, s * 0.38f); lineTo(s * 0.5f, s * 0.58f)
                    lineTo(s * 0.70f, s * 0.38f)
                }, tint, style = st
            )
            drawPath(
                Path().apply {
                    moveTo(s * 0.12f, s * 0.66f); lineTo(s * 0.12f, s * 0.76f)
                    cubicTo(s * 0.12f, s * 0.85f, s * 0.17f, s * 0.90f, s * 0.26f, s * 0.90f)
                    lineTo(s * 0.74f, s * 0.90f)
                    cubicTo(s * 0.83f, s * 0.90f, s * 0.88f, s * 0.85f, s * 0.88f, s * 0.76f)
                    lineTo(s * 0.88f, s * 0.66f)
                }, tint, style = st
            )
        }
        Glyph.PIN -> {
            drawPath(
                Path().apply {
                    moveTo(s * 0.5f, s * 0.92f)
                    cubicTo(s * 0.5f, s * 0.92f, s * 0.16f, s * 0.60f, s * 0.16f, s * 0.40f)
                    cubicTo(s * 0.16f, s * 0.21f, s * 0.31f, s * 0.08f, s * 0.5f, s * 0.08f)
                    cubicTo(s * 0.69f, s * 0.08f, s * 0.84f, s * 0.21f, s * 0.84f, s * 0.40f)
                    cubicTo(s * 0.84f, s * 0.60f, s * 0.5f, s * 0.92f, s * 0.5f, s * 0.92f)
                    close()
                }, tint
            )
            drawCircle(Ink.Panel, radius = s * 0.13f, center = Offset(s / 2, s * 0.38f))
        }
    }
}

/** A press that answers back: everything tappable dips slightly under the finger. */
@Composable
private fun Modifier.pressable(
    enabled: Boolean,
    shape: androidx.compose.ui.graphics.Shape,
    onClick: () -> Unit,
): Modifier {
    val interaction = remember { MutableInteractionSource() }
    val pressed by interaction.collectIsPressedAsState()
    val scale by animateFloatAsState(
        if (pressed && enabled) 0.968f else 1f,
        spring(Spring.DampingRatioMediumBouncy, Spring.StiffnessHigh), label = "press"
    )
    return this
        .scale(scale)
        .clip(shape)
        .clickable(
            enabled = enabled, interactionSource = interaction,
            indication = ripple(color = Color.White), onClick = onClick
        )
}

@Composable
fun PrimaryButton(
    label: String,
    glyph: Glyph?,
    enabled: Boolean,
    modifier: Modifier = Modifier,
    onClick: () -> Unit,
) {
    val shape = RoundedCornerShape(50)
    val bg = if (enabled) Ink.Accent else Ink.PanelHi
    val fg = if (enabled) Color(0xFF05121F) else Ink.TextFaint
    Box(
        modifier
            .height(54.dp)
            .pressable(enabled, shape, onClick)
            .background(bg, shape),
        contentAlignment = Alignment.Center
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(9.dp)
        ) {
            glyph?.let { Icon(it, fg, 19) }
            Text(label, color = fg, style = MaterialTheme.typography.labelLarge)
        }
    }
}

@Composable
fun SecondaryButton(
    label: String,
    glyph: Glyph?,
    enabled: Boolean,
    modifier: Modifier = Modifier,
    onClick: () -> Unit,
) {
    val shape = RoundedCornerShape(50)
    val fg = if (enabled) Ink.Text else Ink.TextFaint
    Box(
        modifier
            .height(54.dp)
            .pressable(enabled, shape, onClick)
            .background(Ink.PanelHi, shape)
            .border(1.dp, Ink.Hairline, shape),
        contentAlignment = Alignment.Center
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(9.dp)
        ) {
            glyph?.let { Icon(it, fg, 19) }
            Text(label, color = fg, style = MaterialTheme.typography.labelLarge)
        }
    }
}

/** Circular control that floats over the map. */
@Composable
fun MapButton(glyph: Glyph, tint: Color = Ink.Text, onClick: () -> Unit) {
    Box(
        Modifier
            .size(46.dp)
            .pressable(true, CircleShape, onClick)
            .background(Ink.Panel.copy(alpha = 0.92f), CircleShape)
            .border(1.dp, Ink.Hairline, CircleShape),
        contentAlignment = Alignment.Center
    ) { Icon(glyph, tint, 21) }
}

/** One measurement, in a shape that keeps a row of them evenly spaced. */
@Composable
fun StatChip(value: String, label: String, modifier: Modifier = Modifier) {
    Column(
        modifier
            .clip(RoundedCornerShape(13.dp))
            .background(Ink.PanelSoft)
            .padding(horizontal = 9.dp, vertical = 9.dp),
        verticalArrangement = Arrangement.spacedBy(2.dp)
    ) {
        Text(value, style = Num.copy(fontSize = 14.sp, color = Ink.Text),
            maxLines = 1, softWrap = false)
        Text(label, color = Ink.TextFaint, style = MaterialTheme.typography.labelSmall,
            maxLines = 1, softWrap = false)
    }
}

// ---- the waiting state ----------------------------------------------------------------

/**
 * The wait, built out of what the model is actually doing.
 *
 * The encoder cuts the photograph into a 24x24 grid of 16-pixel patches and attends over them,
 * so the wait is drawn as exactly that: the user's own image, divided, with a wave of attention
 * travelling across it. It is honest about the work rather than a spinner standing in for it,
 * and because the phases have wildly different costs — roughly six seconds to encode, half a
 * second to search — the sweep slows and settles as the encoder finishes.
 */
@Composable
fun WorkingPanel(
    stage: Engine.Stage?,
    elapsedMs: Long,
    thumb: ImageBitmap?,
    modifier: Modifier = Modifier,
) {
    val t = rememberInfiniteTransition(label = "scan")
    val phase by t.animateFloat(
        0f, 1f, infiniteRepeatable(tween(2600, easing = LinearEasing)), label = "phase"
    )
    val settled = stage == Engine.Stage.SEARCHING || stage == Engine.Stage.RANKING
    val settle by animateFloatAsState(
        if (settled) 1f else 0f, tween(500, easing = FastOutSlowInEasing), label = "settle"
    )

    val caption = when (stage) {
        Engine.Stage.PREPARING -> "Reading the photograph"
        Engine.Stage.ENCODING -> "Encoding 576 image patches"
        Engine.Stage.SEARCHING -> "Searching 300k places"
        Engine.Stage.RANKING -> "Ranking candidates"
        null -> ""
    }

    Column(modifier, verticalArrangement = Arrangement.spacedBy(18.dp)) {
        Box(
            Modifier
                .fillMaxWidth()
                .height(172.dp)
                .clip(RoundedCornerShape(20.dp))
                .background(Ink.PanelSoft)
        ) {
            thumb?.let {
                Image(
                    it, null, Modifier.fillMaxSize(),
                    contentScale = ContentScale.Crop,
                    alpha = 0.40f + 0.16f * settle
                )
            }
            Canvas(Modifier.fillMaxSize()) {
                val n = 24
                val cw = size.width / n
                val ch = size.height / n
                for (r in 0 until n) {
                    for (c in 0 until n) {
                        // diagonal wave: cells enter and leave in a travelling band
                        val cellPhase = (r + c) / (2f * (n - 1))
                        var d = phase - cellPhase
                        if (d < 0f) d += 1f
                        val pulse = kotlin.math.exp(-(d / 0.10f) * (d / 0.10f))
                        val a = (0.04f + 0.50f * pulse) * (1f - 0.55f * settle)
                        if (a <= 0.02f) continue
                        drawRect(
                            Ink.Accent.copy(alpha = a * 0.5f),
                            topLeft = Offset(c * cw, r * ch),
                            size = Size(cw - 1f, ch - 1f)
                        )
                    }
                }
                for (i in 1 until n) {
                    val a = 0.05f
                    drawLine(Color.White.copy(alpha = a), Offset(i * cw, 0f), Offset(i * cw, size.height), 0.8f)
                    drawLine(Color.White.copy(alpha = a), Offset(0f, i * ch), Offset(size.width, i * ch), 0.8f)
                }
            }
        }

        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text(caption, color = Ink.Text, style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.weight(1f))
                Text(
                    String.format(java.util.Locale.ROOT, "%.1fs", elapsedMs / 1000.0),
                    style = Num.copy(fontSize = 13.sp, color = Ink.TextDim)
                )
            }
            // an elapsed-time rule: eases toward, but never reaches, the measured typical time
            val frac = (1f - kotlin.math.exp(-(elapsedMs / 1000f) / 3.4f)).coerceIn(0f, 0.97f)
            val width by animateFloatAsState(
                if (settled) 1f else frac, tween(400, easing = LinearEasing), label = "bar"
            )
            Box(
                Modifier.fillMaxWidth().height(4.dp)
                    .clip(RoundedCornerShape(50)).background(Ink.PanelHi)
            ) {
                Box(
                    Modifier.fillMaxWidth(width).height(4.dp)
                        .clip(RoundedCornerShape(50)).background(Ink.Accent)
                )
            }
        }
    }
}

// ---- result ----------------------------------------------------------------------------

@Composable
private fun ConfidenceBadge(confidence: Double, hitKm: Double) {
    val tint = when {
        confidence >= 0.7 -> Ink.Good
        confidence >= 0.4 -> Ink.Warn
        else -> Ink.TextDim
    }
    Row(
        Modifier
            .clip(RoundedCornerShape(50))
            .background(tint.copy(alpha = 0.13f))
            .padding(horizontal = 11.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(7.dp)
    ) {
        Box(Modifier.size(7.dp).clip(CircleShape).background(tint))
        Text(
            "${Math.round(confidence * 100)}% within ${hitKm.toInt()} km",
            style = Num.copy(fontSize = 12.5.sp, color = tint, fontWeight = FontWeight.Medium)
        )
    }
}

@Composable
fun ResultSheet(
    country: String,
    result: LocateResult,
    encMs: Long,
    thumb: ImageBitmap?,
    confidence: Double?,
    hitKm: Double,
    onCopy: () -> Unit,
    onPick: () -> Unit,
    onCamera: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(modifier, verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            thumb?.let {
                Image(
                    it, null,
                    Modifier.size(58.dp).clip(RoundedCornerShape(16.dp)),
                    contentScale = ContentScale.Crop
                )
            }
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(7.dp)) {
                Text(
                    country.ifEmpty { "Open water" },
                    color = Ink.Text, style = MaterialTheme.typography.headlineMedium,
                    maxLines = 1
                )
                if (confidence != null) ConfidenceBadge(confidence, hitKm)
                else Caption("Estimated location", Ink.TextDim)
            }
        }

        Row(
            Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(16.dp))
                .background(Ink.PanelSoft)
                .clickable { onCopy() }
                .padding(horizontal = 15.dp, vertical = 13.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
                Caption("Coordinates", Ink.TextFaint)
                Text(
                    // hemispheres rather than minus signs: this is the line people read out
                    formatLatLon(result.lat, result.lon),
                    style = Num.copy(fontSize = 16.sp, color = Ink.Signal,
                        fontWeight = FontWeight.SemiBold),
                    maxLines = 1, softWrap = false
                )
            }
            Icon(Glyph.COPY, Ink.TextDim, 19)
        }

        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            StatChip(String.format(java.util.Locale.ROOT, "%,d", result.nCandidates),
                "compared", Modifier.weight(1f))
            StatChip("${result.gateCells}", "regions", Modifier.weight(1f))
            StatChip(String.format(java.util.Locale.ROOT, "%.1fs", encMs / 1000.0),
                "encode", Modifier.weight(1f))
            StatChip("${result.msScore + result.msRerank} ms", "search", Modifier.weight(1f))
        }

        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            PrimaryButton("New photo", Glyph.PHOTO, true, Modifier.weight(1f)) { onPick() }
            SecondaryButton("Camera", Glyph.CAMERA, true, Modifier.width(132.dp)) { onCamera() }
        }
    }
}

@Composable
fun StartSheet(ready: Boolean, onPick: () -> Unit, onCamera: () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(18.dp)) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(
                "Where was this taken?",
                color = Ink.Text, style = MaterialTheme.typography.headlineMedium
            )
            Caption(
                if (ready) "Pick a photograph and it lands on the map. No connection needed."
                else "Loading the index…"
            )
        }
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            PrimaryButton("Choose photo", Glyph.PHOTO, ready, Modifier.weight(1f)) { onPick() }
            SecondaryButton("Camera", Glyph.CAMERA, ready, Modifier.width(132.dp)) { onCamera() }
        }
    }
}

// ---- the one-time download -------------------------------------------------------------

private fun gb(b: Long) = String.format(java.util.Locale.ROOT, "%.2f", b / 1e9)

/** What a fresh install shows instead of the start sheet: the app asking for its brain. */
@Composable
fun DownloadOfferSheet(totalBytes: Long, onStart: () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(18.dp)) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(
                "One download, then fully offline",
                color = Ink.Text, style = MaterialTheme.typography.headlineMedium
            )
            Caption(
                "Everything runs on this phone — the neural encoder and an index of 300,000 " +
                        "places. They are a ${gb(totalBytes)} GB download, needed once. " +
                        "Wi-Fi recommended."
            )
        }
        PrimaryButton(
            "Download ${gb(totalBytes)} GB", Glyph.DOWNLOAD, true, Modifier.fillMaxWidth()
        ) { onStart() }
    }
}

@Composable
fun DownloadProgressSheet(p: FetchProgress) {
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text(
                "Downloading", color = Ink.Text,
                style = MaterialTheme.typography.titleMedium
            )
            Spacer(Modifier.weight(1f))
            if (p.bytesPerSec > 1e4) Text(
                String.format(java.util.Locale.ROOT, "%.0f MB/s", p.bytesPerSec / 1e6),
                style = Num.copy(fontSize = 13.sp, color = Ink.TextDim)
            )
        }
        val frac = (p.doneBytes.toFloat() / p.totalBytes).coerceIn(0f, 1f)
        Box(
            Modifier.fillMaxWidth().height(4.dp)
                .clip(RoundedCornerShape(50)).background(Ink.PanelHi)
        ) {
            Box(
                Modifier.fillMaxWidth(frac).height(4.dp)
                    .clip(RoundedCornerShape(50)).background(Ink.Accent)
            )
        }
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text(
                "${gb(p.doneBytes)} of ${gb(p.totalBytes)} GB · file ${p.fileIndex}/${p.fileCount}",
                style = Num.copy(fontSize = 12.5.sp, color = Ink.TextDim)
            )
            Spacer(Modifier.weight(1f))
            val left = p.totalBytes - p.doneBytes
            if (p.bytesPerSec > 1e4) {
                val s = (left / p.bytesPerSec).toInt()
                Text(
                    if (s < 90) "${s}s left" else "${(s + 30) / 60} min left",
                    style = Num.copy(fontSize = 12.5.sp, color = Ink.TextDim)
                )
            }
        }
        Caption("Runs in the background — lock the phone or leave, it keeps going. " +
                "An interrupted download resumes where it stopped.", Ink.TextFaint)
    }
}

@Composable
fun DownloadFailedSheet(message: String, onRetry: () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(
                "Download interrupted",
                color = Ink.Text, style = MaterialTheme.typography.headlineMedium
            )
            Caption(message, Ink.Warn)
            Caption("Nothing already downloaded is lost — it resumes from where it stopped.")
        }
        PrimaryButton("Resume", Glyph.DOWNLOAD, true, Modifier.fillMaxWidth()) { onRetry() }
    }
}

/**
 * A proper cartographic scale bar — a measured span with end ticks and its length written
 * beside it. A bare number floating in a corner means nothing to anyone.
 */
@Composable
fun ScaleBar(kmPerPx: Double, modifier: Modifier = Modifier) {
    if (kmPerPx <= 0 || kmPerPx.isNaN() || kmPerPx.isInfinite()) return
    val steps = listOf(
        0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0,
        100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0
    )
    val target = 88.0
    val km = steps.firstOrNull { it / kmPerPx >= target } ?: steps.last()
    val px = (km / kmPerPx).coerceAtMost(220.0)
    val label = if (km < 1.0) "${(km * 1000).toInt()} m" else "${km.toInt()} km"
    val density = LocalDensity.current

    Row(
        modifier
            .clip(RoundedCornerShape(50))
            .background(Ink.Base.copy(alpha = 0.72f))
            .padding(horizontal = 11.dp, vertical = 5.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(label, style = Num.copy(fontSize = 11.sp, color = Ink.TextDim))
        Canvas(Modifier.size(with(density) { px.toInt().toDp() }, 8.dp)) {
            val y = size.height - 1f
            drawLine(Ink.TextDim, Offset(0f, y), Offset(size.width, y), strokeWidth = 1.4f)
            drawLine(Ink.TextDim, Offset(0.7f, 0f), Offset(0.7f, y), strokeWidth = 1.4f)
            drawLine(Ink.TextDim, Offset(size.width - 0.7f, 0f),
                Offset(size.width - 0.7f, y), strokeWidth = 1.4f)
        }
    }
}

/**
 * The bottom sheet.
 *
 * Rounded at the top and lifted off the terrain by a soft gradient, so the map appears to run
 * underneath it rather than stopping at a hard edge.
 */
@Composable
fun Sheet(visible: Boolean, content: @Composable () -> Unit) {
    AnimatedVisibility(
        visible = visible,
        enter = slideInVertically(spring(Spring.DampingRatioLowBouncy, Spring.StiffnessMediumLow)) { it } +
                fadeIn(tween(220)),
        exit = slideOutVertically(tween(220, easing = FastOutSlowInEasing)) { it } + fadeOut(tween(140)),
    ) {
        Column(Modifier.fillMaxWidth()) {
            Box(
                Modifier.fillMaxWidth().height(26.dp).background(
                    Brush.verticalGradient(listOf(Color.Transparent, Ink.Base.copy(alpha = 0.55f)))
                )
            )
            Column(
                Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(topStart = 28.dp, topEnd = 28.dp))
                    .background(Ink.Panel)
            ) {
                Box(Modifier.fillMaxWidth().padding(22.dp, 22.dp, 22.dp, 20.dp)) { content() }
            }
        }
    }
}
