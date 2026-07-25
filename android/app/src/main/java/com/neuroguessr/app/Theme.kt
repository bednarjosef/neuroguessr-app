package com.neuroguessr.app

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontVariation
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

/**
 * Neutral greys and exactly one signal colour.
 *
 * Two accents fighting each other is what makes an interface look decorative; here the only
 * saturated thing on screen is the answer itself, so the eye goes straight to it and the
 * chrome stays out of the way.
 */
object Ink {
    val Base = Color(0xFF0A0A0B)
    val Panel = Color(0xFF121214)
    val PanelHi = Color(0xFF17171A)
    val Hairline = Color(0xFF26262C)
    val HairSoft = Color(0xFF1B1B20)

    val Text = Color(0xFFEDEDEF)
    val TextDim = Color(0xFF9A9AA2)
    val TextFaint = Color(0xFF63636B)

    /** The prediction. Nothing else in the app is allowed to use it. */
    val Signal = Color(0xFFE4573D)
    val Ghost = Color(0xFF4A4A52)

    val Water = Color(0xFF0C0D10)
    val Land = Color(0xFF1F2024)
    val LandHi = Color(0xFF272930)
    val Border = Color(0xFF44464F)
    val Coast = Color(0xFF4C4F59)
    val Grid = Color(0xFF17181C)
}

// IBM Plex Sans ships as a variable font; picking weights from it needs the opt-in.
@OptIn(androidx.compose.ui.text.ExperimentalTextApi::class)
private val PlexSans = FontFamily(
    Font(R.font.ibmplexsans, FontWeight.Normal,
        variationSettings = FontVariation.Settings(FontVariation.weight(400))),
    Font(R.font.ibmplexsans, FontWeight.Medium,
        variationSettings = FontVariation.Settings(FontVariation.weight(500))),
    Font(R.font.ibmplexsans, FontWeight.SemiBold,
        variationSettings = FontVariation.Settings(FontVariation.weight(600))),
)

val Mono = FontFamily(
    Font(R.font.ibmplexmono_regular, FontWeight.Normal),
    Font(R.font.ibmplexmono_medium, FontWeight.Medium),
    Font(R.font.ibmplexmono_semibold, FontWeight.SemiBold),
)

private val Scheme = darkColorScheme(
    primary = Ink.Signal,
    onPrimary = Ink.Base,
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

private fun base(size: Int, weight: FontWeight, tracking: Double = 0.0, line: Int = 0) = TextStyle(
    fontFamily = PlexSans, fontSize = size.sp, fontWeight = weight,
    letterSpacing = tracking.sp, lineHeight = (if (line > 0) line else (size * 1.35).toInt()).sp
)

private val Type = Typography(
    headlineLarge = base(30, FontWeight.Normal, -0.4),
    headlineMedium = base(24, FontWeight.Normal, -0.2),
    titleLarge = base(19, FontWeight.Medium),
    titleMedium = base(15, FontWeight.Medium),
    bodyLarge = base(15, FontWeight.Normal),
    bodyMedium = base(14, FontWeight.Normal),
    bodySmall = base(12, FontWeight.Normal),
    labelLarge = base(13, FontWeight.Medium, 0.6),
    labelMedium = base(11, FontWeight.Medium, 0.9),
    labelSmall = base(10, FontWeight.Medium, 1.1),
)

@Composable
fun NeuroTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = Scheme, typography = Type, content = content)
}
