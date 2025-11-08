package com.example.mobileapp.ui.main

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.draw.clip
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import coil.compose.rememberAsyncImagePainter
import androidx.camera.view.PreviewView
import com.example.mobileapp.ui.theme.MobileAppTheme
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraCaptureActivity : ComponentActivity() {

    companion object {
        private const val TAG = "CameraCaptureActivity"
        const val EXTRA_PHOTO_URI = "photo_uri"
    }

    private var imageCapture: ImageCapture? = null
    private var cameraExecutor: ExecutorService? = null

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                startCameraInternal()
            } else {
                Toast.makeText(this, "Camera permission is required to take photos", Toast.LENGTH_LONG).show()
                finish()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        cameraExecutor = Executors.newSingleThreadExecutor()

        setContent {
            MobileAppTheme {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    CameraCaptureScreen(
                        onCapture = { takePhoto() },
                        onAccept = { uri ->
                            val intent = Intent().apply { putExtra(EXTRA_PHOTO_URI, uri.toString()) }
                            setResult(Activity.RESULT_OK, intent)
                            finish()
                        },
                        onRetake = {
                            // simply discard the saved photo and return to preview state
                            // we keep camera running
                            tempPhotoUri = null
                        }
                    )
                }
            }
        }

        // request camera permission (if not granted)
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCameraInternal()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            cameraExecutor?.shutdown()
        } catch (e: Exception) {
            Log.w(TAG, "Error shutting down camera executor", e)
        }
    }

    // UI state shared with Composables
    // Use a simple mutableState to share state between Composable and Activity functions.
    // tempPhotoUri holds the last captured photo file uri (or null when in preview mode)
    private var tempPhotoUri: Uri? by mutableStateOf(null)
    private var isSaving: Boolean by mutableStateOf(false)

    // Start camera and bind preview + imageCapture
    private fun startCameraInternal() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()

                // Preview
                val preview = Preview.Builder().build()

                // ImageCapture
                imageCapture = ImageCapture.Builder()
                    .setTargetRotation(windowManager.defaultDisplay.rotation)
                    .build()

                // Select back camera as default
                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                // Unbind use cases before rebinding
                cameraProvider.unbindAll()
                // Bind to lifecycle
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture)

                // Provide a way for the composable PreviewView to set its surface provider
                // We can't directly set the surface provider here because PreviewView is created in compose.
                // Instead we will set it from composable through a callback.
                previewProviderRef?.invoke(preview)

            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // callback that composable will set with a Preview to attach its surface later
    private var previewProviderRef: ((preview: Preview) -> Unit)? = null

    // Called from UI (Composables) when user presses Capture
    private fun takePhoto() {
        val imageCapture = imageCapture ?: run {
            Toast.makeText(this, "Camera not ready", Toast.LENGTH_SHORT).show()
            return
        }

        // create output file
        val photoFile = createImageFile()
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        isSaving = true
        imageCapture.takePicture(outputOptions, cameraExecutor!!, object : ImageCapture.OnImageSavedCallback {
            override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                val savedUri = Uri.fromFile(photoFile)
                tempPhotoUri = savedUri
                isSaving = false
            }

            override fun onError(exception: ImageCaptureException) {
                isSaving = false
                Log.e(TAG, "Photo capture failed: ${exception.message}", exception)
                runOnUiThread {
                    Toast.makeText(this@CameraCaptureActivity, "Failed to capture photo: ${exception.message}", Toast.LENGTH_LONG).show()
                }
            }
        })
    }

    private fun createImageFile(): File {
        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val filename = "JPEG_${timeStamp}_.jpg"
        return File(cacheDir, filename)
    }

    // Composable UI
    @Composable
    fun CameraCaptureScreen(
        onCapture: () -> Unit,
        onAccept: (Uri) -> Unit,
        onRetake: () -> Unit
    ) {
        val context = LocalContext.current

        Column(modifier = Modifier.fillMaxSize(), horizontalAlignment = Alignment.CenterHorizontally) {
            // Preview area
            Box(modifier = Modifier
                .fillMaxWidth()
                .height(520.dp), contentAlignment = Alignment.Center) {

                if (tempPhotoUri == null) {
                    // Camera preview (PreviewView inside Compose)
                    AndroidView(factory = { ctx ->
                        val previewView = PreviewView(ctx).apply {
                            scaleType = PreviewView.ScaleType.FILL_CENTER
                        }
                        // set preview surface provider when camera preview is available
                        previewProviderRef = { preview ->
                            preview.setSurfaceProvider(previewView.surfaceProvider)
                        }
                        previewView
                    }, modifier = Modifier.fillMaxSize())
                } else {
                    // show captured image preview
                    Image(
                        painter = rememberAsyncImagePainter(tempPhotoUri),
                        contentDescription = "Captured photo",
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(8.dp)
                            .clip(RoundedCornerShape(8.dp))
                    )
                }

                if (isSaving) {
                    CircularProgressIndicator(modifier = Modifier.align(Alignment.Center))
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Buttons row
            Row(modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp),
                horizontalArrangement = Arrangement.SpaceEvenly) {

                if (tempPhotoUri == null) {
                    // Capture button
                    Button(onClick = { onCapture() }, modifier = Modifier.weight(1f)) {
                        Text("Capture")
                    }
                } else {
                    // Accept / Retake buttons
                    Button(onClick = {
                        tempPhotoUri?.let { uri -> onAccept(uri) }
                    }, modifier = Modifier.weight(1f)) {
                        Text("Accept")
                    }
                    Spacer(modifier = Modifier.width(12.dp))
                    Button(onClick = {
                        // delete temp file if present
                        try {
                            tempPhotoUri?.let { uri ->
                                val f = File(uri.path ?: "")
                                if (f.exists()) f.delete()
                            }
                        } catch (e: Exception) {
                            Log.w(TAG, "Failed to delete temp file", e)
                        }
                        onRetake()
                    }, modifier = Modifier.weight(1f)) {
                        Text("Retake")
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Cancel button
            Button(onClick = { finish() }, modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)) {
                Text("Cancel")
            }
        }
    }
}
