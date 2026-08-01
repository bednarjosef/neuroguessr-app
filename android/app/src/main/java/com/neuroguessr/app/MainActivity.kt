package com.neuroguessr.app

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.math.log2
import kotlin.math.max

class MainActivity : ComponentActivity() {

    private var engine: Engine? = null
    private var worldMap: WorldMap? = null

    /** Set by an adb intent so the full UI flow can be exercised without a human tapping. */
    private val demoImage = mutableStateOf<String?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent { NeuroTheme { AppScreen() } }
        handleIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleIntent(intent)
    }

    // ---- headless verification, driven over adb ---------------------------------------
    private fun handleIntent(intent: Intent?) {
        intent?.getStringExtra("demo")?.let {
            Log.i("NGUI", "demo intent: $it")
            demoImage.value = it
        }
        val selftestDir = intent?.getStringExtra("selftest")
        val benchN = intent?.getIntExtra("bench", 0) ?: 0
        if (selftestDir == null && benchN <= 0) return
        val model = intent?.getStringExtra("model")
        val threads = intent?.getIntExtra("threads", 0) ?: 0
        val ep = intent?.getStringExtra("ep") ?: "cpu"
        lifecycleScope.launch(Dispatchers.Default) {
            runHeadless(selftestDir, benchN, model, threads, ep)
        }
    }

    private fun runHeadless(dir: String?, benchN: Int, model: String?, threads: Int, ep: String) {
        val tag = "NGSELFTEST"
        try {
            engine?.close(); engine = null
            val eng = Engine.load(this, model, threads, ep).also { engine = it }
            Log.i(tag, "engine loaded: ${eng.modelName}, ${eng.cfg.nRows} rows, ${eng.loadMs} ms")
            val map = worldMap ?: WorldMap.fromAssets(this).also { worldMap = it }

            if (dir != null) {
                val files = File(dir).listFiles { f -> f.name.endsWith(".jpg") }?.sortedBy { it.name }
                    ?: emptyList()
                val arr = JSONArray()
                for (f in files) {
                    val bm = decodeOriented(f)
                    val (res, encMs) = eng.locate(bm)
                    arr.put(JSONObject().apply {
                        put("file", f.name); put("lat", res.lat); put("lon", res.lon)
                        put("country", map.countryAt(res.lat, res.lon) ?: "")
                        put("gate_cells", res.gateCells); put("n_candidates", res.nCandidates)
                        put("ms_encode", encMs); put("ms_gate", res.msGate)
                        put("ms_score", res.msScore); put("ms_rerank", res.msRerank)
                        put("top_rows", JSONArray(res.candidates.take(5).map { it.row }))
                    })
                    Log.i(tag, "${f.name} -> ${res.lat},${res.lon} enc=${encMs}ms cand=${res.nCandidates}")
                }
                File(Paths.root(this), "selftest.json").writeText(arr.toString())
                Log.i(tag, "wrote selftest.json (${files.size} images)")
            }

            if (benchN > 0) {
                val f = File(File(Paths.root(this), "testdata"), "val_0000.jpg")
                val bm = if (f.exists()) decodeOriented(f)
                else Bitmap.createBitmap(512, 512, Bitmap.Config.ARGB_8888)
                val enc = LongArray(benchN); val tot = LongArray(benchN)
                for (i in 0 until benchN) {
                    val t = System.currentTimeMillis()
                    val (r, e) = eng.locate(bm)
                    tot[i] = System.currentTimeMillis() - t; enc[i] = e
                    Log.i(tag, "bench iter $i: encode ${e} ms, total ${tot[i]} ms " +
                            "(gate ${r.msGate} score ${r.msScore} rerank ${r.msRerank})")
                }
                enc.sort(); tot.sort()
                Log.i(tag, "BENCH ${eng.modelName}: encode median ${enc[benchN / 2]} ms " +
                        "(min ${enc[0]}) | total median ${tot[benchN / 2]} ms")
            }
            Log.i(tag, "SELFTEST_DONE")
        } catch (e: Throwable) {
            Log.e(tag, "SELFTEST_FAIL ${e::class.java.simpleName}: ${e.message}", e)
        }
    }

    // ---- the app ----------------------------------------------------------------------
    @Composable
    private fun AppScreen() {
        var eng by remember { mutableStateOf<Engine?>(null) }
        var map by remember { mutableStateOf<WorldMap?>(null) }
        var loadError by remember { mutableStateOf<String?>(null) }
        val fetch by DownloadService.state.collectAsState()

        var result by remember { mutableStateOf<LocateResult?>(null) }
        var country by remember { mutableStateOf("") }
        var encMs by remember { mutableStateOf(0L) }
        var stage by remember { mutableStateOf<Engine.Stage?>(null) }
        var elapsed by remember { mutableStateOf(0L) }
        var thumb by remember { mutableStateOf<ImageBitmap?>(null) }

        val camera = remember { MapCamera() }
        val reveal = remember { Animatable(0f) }
        val haptics = LocalHapticFeedback.current
        val scope = rememberCoroutineScope()

        suspend fun loadEngine() = withContext(Dispatchers.IO) {
            val e = Engine.load(this@MainActivity, threads = 4, ep = "xnnpack")
            engine = e
            withContext(Dispatchers.Main) { eng = e }
        }

        LaunchedEffect(Unit) {
            withContext(Dispatchers.IO) {
                try {
                    val m = WorldMap.fromAssets(this@MainActivity)
                    worldMap = m
                    withContext(Dispatchers.Main) { map = m }
                    Paths.ensureDirs(this@MainActivity)
                    val manifest = AssetManifest.fromAssets(this@MainActivity)
                    if (AssetDownloader.complete(this@MainActivity, manifest)) loadEngine()
                    // a service run may already be underway (or finished) from a previous
                    // visit — only offer the download when nothing else is happening
                    else DownloadService.state.compareAndSet(
                        null, FetchState.Idle(manifest.totalBytes)
                    )
                } catch (t: Throwable) {
                    withContext(Dispatchers.Main) { loadError = t.message ?: t.toString() }
                }
            }
        }

        // the service finished (possibly while the app was backgrounded): load and play
        LaunchedEffect(fetch) {
            if (fetch !is FetchState.Complete) return@LaunchedEffect
            DownloadService.state.value = null
            try {
                loadEngine()
            } catch (t: Throwable) {
                withContext(Dispatchers.Main) { loadError = t.message ?: t.toString() }
            }
        }

        val notifPerm = rememberLauncherForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { }
        fun startFetch() {
            if (android.os.Build.VERSION.SDK_INT >= 33 && checkSelfPermission(
                    android.Manifest.permission.POST_NOTIFICATIONS
                ) != android.content.pm.PackageManager.PERMISSION_GRANTED
            ) notifPerm.launch(android.Manifest.permission.POST_NOTIFICATIONS)
            DownloadService.start(this)
        }

        // a live clock while the encoder works, so the wait never looks like a freeze
        LaunchedEffect(stage) {
            if (stage == null) return@LaunchedEffect
            val t0 = System.currentTimeMillis()
            while (isActive) {
                elapsed = System.currentTimeMillis() - t0
                delay(100)
            }
        }

        fun run(bm: Bitmap) {
            val e = eng ?: return
            thumb = bm.asImageBitmap()
            result = null
            stage = Engine.Stage.PREPARING
            scope.launch {
                reveal.snapTo(0f)
                // pull back to a wide view while it thinks, so the reveal has somewhere to
                // fly in from and the answer arrives as a movement rather than a jump cut
                camera.launchAnimation(this) { zoomTo(camera.minZoomLog + 0.1f, 1100) }
                val out = withContext(Dispatchers.Default) {
                    e.locate(bm) { s -> stage = s }
                }
                val r = out.first
                encMs = out.second
                country = map?.countryAt(r.lat, r.lon) ?: ""
                stage = null
                result = r
                haptics.performHapticFeedback(HapticFeedbackType.LongPress)
                camera.launchAnimation(scope) {
                    flyTo(r.lon.toFloat(), r.lat.toFloat(), spreadZoomLog(r, camera), 950)
                }
                reveal.animateTo(1f, spring(Spring.DampingRatioMediumBouncy, Spring.StiffnessLow))
            }
        }

        val pick = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri ?: return@rememberLauncherForActivityResult
            scope.launch(Dispatchers.IO) {
                val tmp = File(cacheDir, "picked.jpg")
                contentResolver.openInputStream(uri)?.use { i -> tmp.outputStream().use { i.copyTo(it) } }
                val bm = decodeOriented(tmp)
                withContext(Dispatchers.Main) { run(bm) }
            }
        }
        val capture = rememberLauncherForActivityResult(
            ActivityResultContracts.TakePicturePreview()
        ) { bm -> if (bm != null) run(bm) }

        val demo = demoImage.value
        LaunchedEffect(demo, eng) {
            if (demo == null || eng == null) return@LaunchedEffect
            demoImage.value = null
            // clearing the flag restarts this effect, so the work has to run somewhere that
            // a key change cannot cancel
            scope.launch {
                val f = File(demo)
                Log.i("NGUI", "demo running on ${f.name} exists=${f.exists()}")
                if (f.exists()) run(withContext(Dispatchers.IO) { decodeOriented(f) })
            }
        }

        Surface(Modifier.fillMaxSize(), color = Ink.Base) {
            Box(Modifier.fillMaxSize()) {
                map?.let { m ->
                    MapCanvas(
                        map = m, camera = camera,
                        pin = result?.let { it.lat to it.lon },
                        cluster = result?.candidates?.take(40) ?: emptyList(),
                        pinReveal = reveal.value,
                        modifier = Modifier.fillMaxSize()
                    )
                }

                // just enough scrim to keep the wordmark legible over bright terrain
                Box(
                    Modifier.fillMaxWidth().height(140.dp).background(
                        Brush.verticalGradient(listOf(Ink.Base.copy(alpha = 0.72f), Color.Transparent))
                    )
                )

                // the wordmark rides on the map as a floating chip rather than in a bar, so
                // nothing steals a strip of the terrain that the map could be using
                Row(
                    Modifier.align(Alignment.TopStart).statusBarsPadding()
                        .padding(start = 16.dp, top = 12.dp)
                        .clip(RoundedCornerShape(50))
                        .background(Ink.Panel.copy(alpha = 0.86f))
                        .border(1.dp, Ink.Hairline, RoundedCornerShape(50))
                        .padding(start = 12.dp, end = 16.dp, top = 8.dp, bottom = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Icon(Glyph.PIN, Ink.Signal, 17)
                    Text(
                        "Neuroguessr", color = Ink.Text,
                        style = MaterialTheme.typography.titleMedium
                    )
                    if (eng == null && loadError == null) {
                        Box(Modifier.size(6.dp).clip(CircleShape).background(Ink.Accent))
                    }
                }

                Column(
                    Modifier.align(Alignment.BottomCenter).fillMaxWidth(),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    // map furniture rides directly above the sheet edge, whatever its height
                    Row(
                        Modifier.fillMaxWidth().padding(start = 16.dp, end = 16.dp, bottom = 12.dp),
                        verticalAlignment = Alignment.Bottom
                    ) {
                        val ppu = camera.pxPerUnit(camera.viewW)
                        val kmPerPx = 111.32 *
                                kotlin.math.cos(Math.toRadians(camera.centerLat.toDouble())) / ppu
                        ScaleBar(kmPerPx, Modifier.padding(bottom = 6.dp))
                        Spacer(Modifier.weight(1f))
                        if (result != null) {
                            MapButton(Glyph.TARGET, Ink.Signal) {
                                result?.let { r ->
                                    camera.launchAnimation(scope) {
                                        flyTo(
                                            r.lon.toFloat(), r.lat.toFloat(),
                                            spreadZoomLog(r, camera), 700
                                        )
                                    }
                                }
                            }
                        }
                    }
                    loadError?.let {
                        Sheet(true) {
                            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                Text("Cannot start", color = Ink.Text,
                                    style = MaterialTheme.typography.titleLarge)
                                Text(it, style = Num.copy(fontSize = 13.sp, color = Ink.TextDim))
                            }
                        }
                    }

                    Sheet(fetch != null && fetch !is FetchState.Complete) {
                        when (val f = fetch) {
                            is FetchState.Idle ->
                                DownloadOfferSheet(f.totalBytes) { startFetch() }
                            is FetchState.Running -> DownloadProgressSheet(f.p)
                            is FetchState.Failed ->
                                DownloadFailedSheet(f.message) { startFetch() }
                            else -> {}
                        }
                    }

                    Sheet(stage != null) {
                        WorkingPanel(stage, elapsed, thumb, Modifier.fillMaxWidth())
                    }

                    Sheet(result != null && stage == null) {
                        result?.let { r ->
                            ResultSheet(
                                country, r, encMs, thumb,
                                confidence = if (eng?.cfg?.hasConfidence == true) r.confidence else null,
                                hitKm = eng?.cfg?.confHitKm ?: 25.0,
                                onCopy = {
                                    val cm = getSystemService(android.content.ClipboardManager::class.java)
                                    cm.setPrimaryClip(
                                        android.content.ClipData.newPlainText(
                                            "coordinates",
                                            String.format(java.util.Locale.ROOT, "%.5f, %.5f", r.lat, r.lon)
                                        )
                                    )
                                    haptics.performHapticFeedback(HapticFeedbackType.TextHandleMove)
                                },
                                onPick = { pick.launch("image/*") },
                                onCamera = { capture.launch(null) },
                            )
                        }
                    }

                    Sheet(result == null && stage == null && loadError == null && fetch == null) {
                        StartSheet(
                            ready = eng != null,
                            onPick = { pick.launch("image/*") },
                            onCamera = { capture.launch(null) },
                        )
                    }
                    Box(Modifier.fillMaxWidth().background(Ink.Panel)) {
                        Spacer(Modifier.navigationBarsPadding().fillMaxWidth())
                    }
                }
            }
        }
    }
}

/**
 * Frame the answer with its surroundings.
 *
 * Zooming to the candidate cluster alone lands you on an unlabelled patch of road with no
 * idea which part of the world you are looking at, so the view never goes tighter than about
 * 300 km across — enough to show the region, borders and coastline that give the pin meaning.
 * A scattered cluster still pulls the camera back further.
 */
private const val MIN_SPAN_DEG = 3.6      // ~300 km of screen width; the closest we ever go
private const val MAX_SPAN_DEG = 120.0    // continental, for a search that agreed on nothing

private fun spreadZoomLog(r: LocateResult, camera: MapCamera): Float {
    val top = r.candidates.take(12)
    var span = 0.0
    for (c in top) {
        span = max(span, kotlin.math.abs(c.lat - r.lat))
        span = max(span, kotlin.math.abs(c.lon - r.lon) *
                kotlin.math.cos(Math.toRadians(r.lat)))
    }
    val degrees = (span * 4).coerceIn(MIN_SPAN_DEG, MAX_SPAN_DEG)
    // Mercator stretches everything away from the equator, so a Finnish result needs less
    // scale than a Kenyan one to frame the same ground distance
    val stretch = 1.0 / kotlin.math.cos(Math.toRadians(r.lat)).coerceAtLeast(0.15)
    return camera.clampZoom(log2(360.0 / (degrees * stretch)).toFloat())
}
