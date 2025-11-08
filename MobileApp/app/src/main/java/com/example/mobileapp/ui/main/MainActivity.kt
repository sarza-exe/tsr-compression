package com.example.mobileapp.ui.main

import android.app.Activity
import android.content.ContentResolver
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.net.toUri
import androidx.lifecycle.lifecycleScope
import coil.compose.rememberAsyncImagePainter
import com.example.mobileapp.ui.theme.MobileAppTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

class MainActivity : ComponentActivity() {

    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let { handleImageUri(it) }
    }

    // Zakladamy ze CameraCaptureActivity zwraca Intent z "photo_uri" (String)
    private val cameraLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            val data = result.data
            val uriString = data?.getStringExtra("photo_uri")
            if (!uriString.isNullOrEmpty()) {
                handleImageUri(Uri.parse(uriString))
            } else {
                Toast.makeText(this, "No photo returned", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // stan UI
    private var selectedImageUri by mutableStateOf<Uri?>(null)
    private var isProcessing by mutableStateOf(false)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            MobileAppTheme {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    MainScreen(
                        selectedImageUri = selectedImageUri,
                        isProcessing = isProcessing,
                        onPickFromGallery = { pickImageLauncher.launch("image/*") },
                        onOpenCamera = {
                            // uruchom CameraCaptureActivity
                            val intent = Intent(this, CameraCaptureActivity::class.java)
                            cameraLauncher.launch(intent)
                        },
                        onProcessSelected = {
                            selectedImageUri?.let { uri ->
                                startDetectionFlow(uri)
                            } ?: run {
                                Toast.makeText(this@MainActivity, "No image selected", Toast.LENGTH_SHORT).show()
                            }
                        }
                    )
                }
            }
        }
    }

    private fun handleImageUri(uri: Uri) {
        // ustaw podgląd i automatycznie uruchom przetwarzanie (albo poczekaj na przycisk: tutaj automatycznie)
        selectedImageUri = uri
        startDetectionFlow(uri)
    }

    private fun startDetectionFlow(uri: Uri) {
        // Nie blokuj UI thread
        lifecycleScope.launch {
            isProcessing = true
            val bmp = withContext(Dispatchers.IO) { decodeBitmapFromUri(this@MainActivity.contentResolver, uri, maxDim = 1280) }
            if (bmp == null) {
                Toast.makeText(this@MainActivity, "Nie można wczytać obrazu", Toast.LENGTH_SHORT).show()
                isProcessing = false
                return@launch
            }

            // W tym miejscu próbujemy uruchomić detekcję YOLO.
            // mamy  klasę YoloDetector z metodą detect(bitmap): List<Detection>
            try {
                val detector = com.example.mobileapp.inference.YoloDetector(applicationContext)
                val detections = withContext(Dispatchers.Default) {
                    // dostosowac inputSize/confThreshold jeśli trzeba
                    detector.detect(bmp)
                }

                if (detections.isEmpty()) {
                    Toast.makeText(this@MainActivity, "No sign found on the photo", Toast.LENGTH_LONG).show()
                    isProcessing = false
                    return@launch
                }

                // wybierz detekcję z najwyższym score
                val best = detections.maxByOrNull { it.score }!!

                // przelicz współrzędne jeśli detekcje były w skali inputSize -> tutaj zakładamy że są w pikselach oryginalnego obrazu
                val cropped = cropBitmapSafe(bmp, best.x1, best.y1, best.x2, best.y2)

                // zapisz wycięty znak do cache i przekieruj do ResultActivity
                val savedUri = withContext(Dispatchers.IO) { saveBitmapToCache(this@MainActivity, cropped) }
                if (savedUri == null) {
                    Toast.makeText(this@MainActivity, "Failed to save cropped sign", Toast.LENGTH_SHORT).show()
                    isProcessing = false
                    return@launch
                }

                // uruchom ResultActivity
                val intent = Intent(this@MainActivity, ResultActivity::class.java).apply {
                    putExtra("crop_uri", savedUri.toString())
                }
                startActivity(intent)

            } catch (e: ClassNotFoundException) {
                // YoloDetector nie istnieje jeszcze w projekcie
                Toast.makeText(this@MainActivity, "YOLO detector not available yet. Add com.example.mobileapp.inference.YoloDetector", Toast.LENGTH_LONG).show()
            } catch (e: Throwable) {
                // inny błąd podczas detekcji
                // an operation is not impelemented rzuca yolodetector.kt
                e.printStackTrace()
                Toast.makeText(this@MainActivity, "Error during detection: ${e.localizedMessage}", Toast.LENGTH_LONG).show()
            } finally {
                isProcessing = false
            }
        }
    }

    // --- pomocnicze funkcje ---

    private fun cropBitmapSafe(bitmap: Bitmap, xf: Float, yf: Float, x2f: Float, y2f: Float): Bitmap {
        val left = xf.toInt().coerceIn(0, bitmap.width - 1)
        val top = yf.toInt().coerceIn(0, bitmap.height - 1)
        val right = x2f.toInt().coerceIn(left + 1, bitmap.width)
        val bottom = y2f.toInt().coerceIn(top + 1, bitmap.height)
        val width = right - left
        val height = bottom - top
        return Bitmap.createBitmap(bitmap, left, top, width, height)
    }

    private fun saveBitmapToCache(context: Context, bitmap: Bitmap): Uri? {
        return try {
            val filename = "crop_${System.currentTimeMillis()}.jpg"
            val file = File(context.cacheDir, filename)
            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
                out.flush()
            }
            file.toUri()
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    private fun decodeBitmapFromUri(contentResolver: ContentResolver, uri: Uri, maxDim: Int = 1280): Bitmap? {
        try {
            // Najpierw odczyt rozmiarów
            val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            var input: InputStream? = null
            try {
                input = contentResolver.openInputStream(uri)
                BitmapFactory.decodeStream(input, null, options)
            } finally {
                input?.close()
            }

            // oblicz inSampleSize
            val (width, height) = options.outWidth to options.outHeight
            var inSampleSize = 1
            val largest = maxOf(width, height)
            if (largest > maxDim) {
                inSampleSize = 1
                while (largest / inSampleSize > maxDim) {
                    inSampleSize *= 2
                }
            }

            val decodeOptions = BitmapFactory.Options().apply { this.inSampleSize = inSampleSize }
            var decoded: Bitmap? = null
            var input2: InputStream? = null
            try {
                input2 = contentResolver.openInputStream(uri)
                decoded = BitmapFactory.decodeStream(input2, null, decodeOptions)
            } finally {
                input2?.close()
            }
            return decoded
        } catch (e: Exception) {
            e.printStackTrace()
            return null
        }
    }
}

@Composable
fun MainScreen(
    selectedImageUri: Uri?,
    isProcessing: Boolean,
    onPickFromGallery: () -> Unit,
    onOpenCamera: () -> Unit,
    onProcessSelected: () -> Unit
) {
    val context = LocalContext.current

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(modifier = Modifier.height(24.dp))
        Text(text = "Traffic Sign Tester", style = MaterialTheme.typography.headlineSmall)
        Spacer(modifier = Modifier.height(24.dp))

        // Buttons
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
            Button(onClick = onPickFromGallery, modifier = Modifier.weight(1f)) {
                Text("Recognise sign from gallery")
            }
            Spacer(modifier = Modifier.width(12.dp))
            Button(onClick = onOpenCamera, modifier = Modifier.weight(1f)) {
                Text("Recognise sign from photo")
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        // podgląd wybranego obrazka
        if (selectedImageUri != null) {
            Box(modifier = Modifier
                .fillMaxWidth()
                .height(300.dp),
                contentAlignment = Alignment.Center
            ) {
                Image(
                    painter = rememberAsyncImagePainter(selectedImageUri),
                    contentDescription = "Selected image",
                    modifier = Modifier
                        .fillMaxSize()
                        .clip(RoundedCornerShape(8.dp))
                )
            }

            Spacer(modifier = Modifier.height(12.dp))
            Button(onClick = onProcessSelected) {
                Text("Process image")
            }
        } else {
            Text("No image selected", style = MaterialTheme.typography.bodyMedium)
        }

        Spacer(modifier = Modifier.height(24.dp))

        if (isProcessing) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(modifier = Modifier.size(36.dp))
                Spacer(modifier = Modifier.width(12.dp))
                Text("Processing...", style = MaterialTheme.typography.bodyMedium)
            }
        }
    }
}