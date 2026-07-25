package com.neuroguessr.app

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
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
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
fun Hairline(modifier: Modifier = Modifier, color: Color = Ink.HairSoft) =
    Box(modifier.fillMaxWidth().height(1.dp).background(color))

/** Field names are set small and tracked out, so hierarchy comes from type, not from boxes. */
@Composable
fun FieldLabel(text: String, color: Color = Ink.TextFaint, modifier: Modifier = Modifier) {
    Text(text.uppercase(), color = color, style = MaterialTheme.typography.labelSmall,
        modifier = modifier)
}

/** Flat, hairline-bordered, uppercase. No fills, no glow, no rounded pill. */
@Composable
fun ActionButton(
    label: String,
    enabled: Boolean,
    modifier: Modifier = Modifier,
    emphasis: Boolean = false,
    onClick: () -> Unit,
) {
    val border = when {
        !enabled -> Ink.HairSoft
        emphasis -> Ink.TextDim
        else -> Ink.Hairline
    }
    val fg = when {
        !enabled -> Ink.TextFaint
        emphasis -> Ink.Text
        else -> Ink.TextDim
    }
    Box(
        modifier
            .height(46.dp)
            .clip(RoundedCornerShape(3.dp))
            .background(if (emphasis && enabled) Ink.PanelHi else Color.Transparent)
            .border(1.dp, border, RoundedCornerShape(3.dp))
            .clickable(enabled = enabled) { onClick() },
        contentAlignment = Alignment.Center
    ) {
        Text(label.uppercase(), color = fg, style = MaterialTheme.typography.labelMedium)
    }
}

@Composable
fun Readout(
    label: String,
    value: String,
    modifier: Modifier = Modifier,
    tint: Color = Ink.Text,
    size: Int = 15,
) {
    Column(modifier, verticalArrangement = Arrangement.spacedBy(5.dp)) {
        FieldLabel(label)
        Text(value, color = tint, fontFamily = Mono, fontSize = size.sp,
            fontWeight = FontWeight.Normal, letterSpacing = 0.sp)
    }
}

/**
 * The waiting state, built out of what the model is actually doing.
 *
 * The encoder cuts the photograph into a 24x24 grid of 16-pixel patches and attends over
 * them, so the wait is drawn as exactly that: the user's own image, divided, with a wave of
 * attention travelling across it. It is honest about the work rather than a spinner standing
 * in for it, and because the phases have wildly different costs — roughly six seconds to
 * encode, half a second to search — the sweep slows and settles as the encoder finishes.
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
    // the sweep is for encoding; once we are searching the grid holds and quietly dims
    val settled = stage == Engine.Stage.SEARCHING || stage == Engine.Stage.RANKING
    val settle by animateFloatAsState(
        if (settled) 1f else 0f, tween(500, easing = FastOutSlowInEasing), label = "settle"
    )

    val caption = when (stage) {
        Engine.Stage.PREPARING -> "Reading photograph"
        Engine.Stage.ENCODING -> "Encoding 576 image patches"
        Engine.Stage.SEARCHING -> "Searching 300k places"
        Engine.Stage.RANKING -> "Ranking candidates"
        null -> ""
    }

    Column(modifier, verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Box(
            Modifier
                .fillMaxWidth()
                .height(168.dp)
                .clip(RoundedCornerShape(3.dp))
                .border(1.dp, Ink.Hairline, RoundedCornerShape(3.dp))
        ) {
            thumb?.let {
                Image(
                    it, null, Modifier.fillMaxSize(),
                    contentScale = ContentScale.Crop,
                    alpha = 0.30f + 0.10f * settle
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
                        val a = (0.05f + 0.55f * pulse) * (1f - 0.55f * settle)
                        if (a <= 0.02f) continue
                        drawRect(
                            Ink.Signal.copy(alpha = a * 0.5f),
                            topLeft = Offset(c * cw, r * ch),
                            size = androidx.compose.ui.geometry.Size(cw - 1f, ch - 1f)
                        )
                    }
                }
                // the grid itself, faint, so the division is legible even between pulses
                for (i in 1 until n) {
                    val a = 0.06f
                    drawLine(Ink.Text.copy(alpha = a), Offset(i * cw, 0f), Offset(i * cw, size.height), 0.8f)
                    drawLine(Ink.Text.copy(alpha = a), Offset(0f, i * ch), Offset(size.width, i * ch), 0.8f)
                }
            }
        }

        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text(caption, color = Ink.Text, style = MaterialTheme.typography.bodyMedium)
            Spacer(Modifier.weight(1f))
            Text(
                String.format(java.util.Locale.ROOT, "%.1fs", elapsedMs / 1000.0),
                color = Ink.TextDim, fontFamily = Mono, fontSize = 13.sp
            )
        }

        // an elapsed-time rule: eases toward, but never reaches, the measured typical duration
        val frac = (1f - kotlin.math.exp(-(elapsedMs / 1000f) / 3.4f)).coerceIn(0f, 0.97f)
        Box(Modifier.fillMaxWidth().height(2.dp).background(Ink.HairSoft)) {
            Box(
                Modifier.fillMaxWidth(if (settled) 1f else frac).height(2.dp)
                    .background(Ink.Signal.copy(alpha = 0.75f))
            )
        }
    }
}

@Composable
fun StageProgress(stage: Engine.Stage?, elapsedMs: Long, modifier: Modifier = Modifier) {
    val steps = listOf(
        Engine.Stage.PREPARING to "Reading image",
        Engine.Stage.ENCODING to "Encoding",
        Engine.Stage.SEARCHING to "Searching index",
        Engine.Stage.RANKING to "Ranking",
    )
    val active = steps.indexOfFirst { it.first == stage }.coerceAtLeast(0)
    Column(modifier, verticalArrangement = Arrangement.spacedBy(14.dp)) {
        FieldLabel("Working")
        Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
            steps.forEachIndexed { i, (_, name) ->
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        when {
                            i < active -> "·"
                            i == active -> "▸"
                            else -> " "
                        },
                        color = if (i <= active) Ink.Text else Ink.TextFaint,
                        fontFamily = Mono, fontSize = 13.sp,
                        modifier = Modifier.width(18.dp)
                    )
                    Text(
                        name,
                        color = when {
                            i < active -> Ink.TextFaint
                            i == active -> Ink.Text
                            else -> Ink.TextFaint
                        },
                        style = MaterialTheme.typography.bodyMedium
                    )
                    Spacer(Modifier.weight(1f))
                    if (i == active) {
                        Text(
                            String.format(java.util.Locale.ROOT, "%.1fs", elapsedMs / 1000.0),
                            color = Ink.TextDim, fontFamily = Mono, fontSize = 12.sp
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun ResultSheet(
    country: String,
    result: LocateResult,
    encMs: Long,
    thumb: ImageBitmap?,
    onCopy: () -> Unit,
    onPick: () -> Unit,
    onCamera: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(modifier, verticalArrangement = Arrangement.spacedBy(18.dp)) {
        Row(
            verticalAlignment = Alignment.Top,
            horizontalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            thumb?.let {
                Image(
                    it, null,
                    Modifier.size(62.dp).clip(RoundedCornerShape(2.dp))
                        .border(1.dp, Ink.Hairline, RoundedCornerShape(2.dp)),
                    contentScale = ContentScale.Crop
                )
            }
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                FieldLabel("Estimated location")
                Text(
                    country.ifEmpty { "Unnamed territory" },
                    color = Ink.Text, style = MaterialTheme.typography.headlineMedium
                )
            }
        }

        Column(
            Modifier.clickable { onCopy() },
            verticalArrangement = Arrangement.spacedBy(5.dp)
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                FieldLabel("Coordinates")
                Spacer(Modifier.weight(1f))
                Text("copy", color = Ink.TextFaint,
                    style = MaterialTheme.typography.labelSmall)
            }
            Text(
                String.format(java.util.Locale.ROOT, "%.5f  %.5f", result.lat, result.lon),
                color = Ink.Signal, fontFamily = Mono, fontSize = 20.sp,
                fontWeight = FontWeight.Medium
            )
        }

        Hairline()

        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Readout("Candidates", String.format(java.util.Locale.ROOT, "%,d", result.nCandidates), tint = Ink.TextDim, size = 13)
            Readout("Regions", "${result.gateCells}", tint = Ink.TextDim, size = 13)
            Readout("Encode", "${encMs / 1000}.${(encMs % 1000) / 100}s", tint = Ink.TextDim, size = 13)
            Readout("Search", "${result.msScore + result.msRerank}ms", tint = Ink.TextDim, size = 13)
        }

        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            ActionButton("Choose photo", true, Modifier.weight(1f), emphasis = true) { onPick() }
            ActionButton("Camera", true, Modifier.weight(1f)) { onCamera() }
        }
    }
}

@Composable
fun StartSheet(ready: Boolean, onPick: () -> Unit, onCamera: () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(18.dp)) {
        Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
            Text("Locate a photograph",
                color = Ink.Text, style = MaterialTheme.typography.titleLarge)
            if (!ready) {
                Text("Loading index…", color = Ink.TextDim,
                    style = MaterialTheme.typography.bodyMedium)
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            ActionButton("Choose photo", ready, Modifier.weight(1f), emphasis = true) { onPick() }
            ActionButton("Camera", ready, Modifier.weight(1f)) { onCamera() }
        }
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
    val target = 96.0
    val km = steps.firstOrNull { it / kmPerPx >= target } ?: steps.last()
    val px = (km / kmPerPx).coerceAtMost(220.0)
    val label = if (km < 1.0) "${(km * 1000).toInt()} m" else "${km.toInt()} km"
    val density = LocalDensity.current

    Row(modifier, verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Canvas(Modifier.size(with(density) { px.toInt().toDp() }, 9.dp)) {
            val y = size.height / 2f
            drawLine(Ink.TextDim, Offset(0f, y), Offset(size.width, y), strokeWidth = 1.2f)
            drawLine(Ink.TextDim, Offset(0.6f, 0f), Offset(0.6f, size.height), strokeWidth = 1.2f)
            drawLine(Ink.TextDim, Offset(size.width - 0.6f, 0f),
                Offset(size.width - 0.6f, size.height), strokeWidth = 1.2f)
        }
        Text(label, color = Ink.TextDim, fontFamily = Mono, fontSize = 11.sp)
    }
}

@Composable
fun Sheet(visible: Boolean, content: @Composable () -> Unit) {
    AnimatedVisibility(
        visible = visible,
        enter = slideInVertically(tween(300)) { it / 2 } + fadeIn(tween(200)),
        exit = slideOutVertically(tween(200)) { it / 2 } + fadeOut(tween(140)),
    ) {
        Column(Modifier.fillMaxWidth().background(Ink.Panel)) {
            Hairline(color = Ink.Hairline)
            Box(Modifier.padding(22.dp, 20.dp)) { content() }
        }
    }
}
