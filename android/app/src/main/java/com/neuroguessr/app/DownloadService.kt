package com.neuroguessr.app

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch

/**
 * The asset download as a foreground service, so locking the phone or switching apps does not
 * pause the 2.2 GB fetch. Progress flows to whoever is looking: the in-app sheet reads
 * [state], the notification shows the same numbers for everyone else. The service holds a
 * partial wake lock because doze throttles background sockets even for foreground services.
 *
 * If the system kills it anyway, nothing is lost — every finished file is already sealed on
 * disk and the `.part` in flight resumes; reopening the app offers to continue.
 */
class DownloadService : Service() {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var wakeLock: PowerManager.WakeLock? = null
    private var lastNotified = 0L

    override fun onCreate() {
        super.onCreate()
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL, "Asset download", NotificationManager.IMPORTANCE_LOW)
        )
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (!running.compareAndSet(false, true)) return START_NOT_STICKY
        startForeground(NOTIF_ID, notification("Preparing…", 0, 0))
        wakeLock = (getSystemService(POWER_SERVICE) as PowerManager)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "neuroguessr:download")
            .apply { acquire(2 * 60 * 60 * 1000L) }

        scope.launch {
            try {
                val m = AssetManifest.fromAssets(this@DownloadService)
                AssetDownloader.run(this@DownloadService, m) { p ->
                    state.value = FetchState.Running(p)
                    val now = System.currentTimeMillis()
                    if (now - lastNotified > 1000) {
                        lastNotified = now
                        notify(notification(
                            "Downloading — %.2f of %.2f GB".format(
                                p.doneBytes / 1e9, p.totalBytes / 1e9
                            ),
                            p.totalBytes, p.doneBytes,
                        ))
                    }
                }
                state.value = FetchState.Complete
                notify(notification("Ready — open NeuroGuessr", 0, 0, ongoing = false))
            } catch (t: Throwable) {
                if (t is CancellationException) throw t
                Log.w(TAG, "download failed", t)
                state.value = FetchState.Failed(t.message ?: t.toString())
                notify(notification(
                    "Download interrupted — open the app to resume", 0, 0, ongoing = false
                ))
            } finally {
                running.set(false)
                wakeLock?.release(); wakeLock = null
                stopForeground(STOP_FOREGROUND_DETACH)   // leave the final notification up
                stopSelf()
            }
        }
        return START_NOT_STICKY
    }

    private fun notification(
        text: String, max: Long, at: Long, ongoing: Boolean = true,
    ): android.app.Notification {
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE
        )
        return NotificationCompat.Builder(this, CHANNEL)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle("NeuroGuessr")
            .setContentText(text)
            .setContentIntent(open)
            .setOngoing(ongoing)
            .setOnlyAlertOnce(true)
            .apply {
                if (max > 0) setProgress(1000, (at * 1000 / max).toInt(), false)
            }
            .build()
    }

    private fun notify(n: android.app.Notification) =
        getSystemService(NotificationManager::class.java).notify(NOTIF_ID, n)

    override fun onBind(intent: Intent?) = null

    override fun onDestroy() {
        scope.cancel()
        wakeLock?.release(); wakeLock = null
        running.set(false)
        super.onDestroy()
    }

    companion object {
        private const val TAG = "DownloadService"
        private const val CHANNEL = "download"
        private const val NOTIF_ID = 1

        private val running = java.util.concurrent.atomic.AtomicBoolean(false)

        /** The single source of truth the UI renders from, service-owned while it runs. */
        val state = MutableStateFlow<FetchState?>(null)

        fun start(ctx: Context) =
            ContextCompat.startForegroundService(ctx, Intent(ctx, DownloadService::class.java))
    }
}
