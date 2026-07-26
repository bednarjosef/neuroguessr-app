package com.neuroguessr.app

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * A night map, not a void.
 *
 * The surfaces are a deep blue-slate rather than black, so the panels read as lit glass over
 * terrain instead of holes cut in the screen. Two colours do the talking: a cool blue for
 * anything the finger can touch, and a single warm signal reserved for the prediction — the
 * same division of labour every serious map app uses, and the reason the eye always knows
 * which mark is the answer.
 */
object Ink {
    val Base = Color(0xFF080B10)
    val Panel = Color(0xFF111823)
    val PanelHi = Color(0xFF1A2330)
    val PanelSoft = Color(0xFF151E29)
    val Hairline = Color(0xFF243040)
    val HairSoft = Color(0xFF1A2431)

    val Text = Color(0xFFF1F5FA)
    val TextDim = Color(0xFF97A5B8)
    val TextFaint = Color(0xFF61707F)

    /** Anything interactive. */
    val Accent = Color(0xFF5B9CFF)
    val AccentDim = Color(0xFF2E5AA8)
    val AccentWash = Color(0xFF15243D)

    /** The prediction. Nothing else in the app is allowed to use it. */
    val Signal = Color(0xFFFF6A45)
    val Good = Color(0xFF3ECF8E)
    val Warn = Color(0xFFF2B33D)

    val Water = Color(0xFF0B131E)
    val Land = Color(0xFF212C39)
    val LandHi = Color(0xFF26323F)
    val Border = Color(0xFF39485B)
    val Coast = Color(0xFF64748B)
    val Grid = Color(0xFF131C27)
}

private val Sans = FontFamily(
    Font(R.font.inter_regular, FontWeight.Normal),
    Font(R.font.inter_medium, FontWeight.Medium),
    Font(R.font.inter_semibold, FontWeight.SemiBold),
)

/** Inter Display: tighter spacing, drawn for large sizes. Headlines only. */
private val Display = FontFamily(
    Font(R.font.interdisplay_medium, FontWeight.Medium),
    Font(R.font.interdisplay_semibold, FontWeight.SemiBold),
)

/**
 * Figures that do not jitter.
 *
 * A live timer or a coordinate re-lays out on every digit change unless the numerals are
 * tabular, which is what made the old readouts twitch. `tnum` fixes the advance widths
 * without dragging a monospace face into an otherwise humanist interface.
 */
val Num = TextStyle(
    fontFamily = Sans, fontFeatureSettings = "tnum", fontWeight = FontWeight.Medium,
    fontSize = 14.sp, letterSpacing = 0.sp, color = Ink.Text,
)

private val Scheme = darkColorScheme(
    primary = Ink.Accent,
    onPrimary = Color(0xFF06111F),
    primaryContainer = Ink.AccentWash,
    onPrimaryContainer = Ink.Accent,
    secondary = Ink.TextDim,
    background = Ink.Base,
    onBackground = Ink.Text,
    surface = Ink.Panel,
    onSurface = Ink.Text,
    surfaceVariant = Ink.PanelHi,
    onSurfaceVariant = Ink.TextDim,
    outline = Ink.Hairline,
    error = Ink.Signal,
)

private fun t(
    size: Int, weight: FontWeight, tracking: Double = 0.0, line: Double = 1.35,
    family: FontFamily = Sans,
) = TextStyle(
    fontFamily = family, fontSize = size.sp, fontWeight = weight,
    letterSpacing = tracking.sp, lineHeight = (size * line).sp,
)

private val Type = Typography(
    displaySmall = t(34, FontWeight.SemiBold, -1.0, 1.15, Display),
    headlineLarge = t(28, FontWeight.SemiBold, -0.7, 1.2, Display),
    headlineMedium = t(23, FontWeight.SemiBold, -0.5, 1.25, Display),
    titleLarge = t(19, FontWeight.SemiBold, -0.2),
    titleMedium = t(16, FontWeight.Medium, -0.1),
    bodyLarge = t(15, FontWeight.Normal, 0.0, 1.45),
    bodyMedium = t(14, FontWeight.Normal, 0.0, 1.45),
    bodySmall = t(13, FontWeight.Normal, 0.0, 1.4),
    labelLarge = t(15, FontWeight.SemiBold, 0.1),
    labelMedium = t(13, FontWeight.Medium, 0.1),
    labelSmall = t(11, FontWeight.Medium, 0.2),
)

private val Corners = Shapes(
    extraSmall = RoundedCornerShape(8.dp),
    small = RoundedCornerShape(12.dp),
    medium = RoundedCornerShape(16.dp),
    large = RoundedCornerShape(22.dp),
    extraLarge = RoundedCornerShape(28.dp),
)

@Composable
fun NeuroTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = Scheme, typography = Type, shapes = Corners, content = content)
}
