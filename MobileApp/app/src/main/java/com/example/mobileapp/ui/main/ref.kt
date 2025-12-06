//
//
//
//
//
//package com.example.mobileapp.ui.main
//
//import android.graphics.Bitmap
//import android.graphics.BitmapFactory
//import android.net.Uri
//import android.os.Bundle
//import android.widget.Toast
//import androidx.activity.ComponentActivity
//import androidx.activity.compose.setContent
//import androidx.compose.foundation.Image
//import androidx.compose.foundation.background
//import androidx.compose.foundation.layout.*
//import androidx.compose.foundation.shape.RoundedCornerShape
//import androidx.compose.material3.Button
//import androidx.compose.material3.MaterialTheme
//import androidx.compose.material3.Surface
//import androidx.compose.material3.Text
//import androidx.compose.material3.CircularProgressIndicator
//import androidx.compose.runtime.*
//import androidx.compose.ui.Alignment
//import androidx.compose.ui.Modifier
//import androidx.compose.ui.draw.clip
//import androidx.compose.ui.platform.LocalContext
//import androidx.compose.ui.unit.dp
//import androidx.lifecycle.lifecycleScope
//import coil.compose.rememberAsyncImagePainter
//import com.example.mobileapp.inference.Classifier
//import com.example.mobileapp.ui.theme.MobileAppTheme
//import kotlinx.coroutines.Dispatchers
//import kotlinx.coroutines.launch
//import kotlinx.coroutines.withContext
//import java.io.InputStream
//
//class ref : ComponentActivity() {
//
//    companion object {
//        const val EXTRA_CROP_URI = "crop_uri"
//    }
//
//    private var classifier: Classifier? = null
//
//    override fun onCreate(savedInstanceState: Bundle?) {
//        super.onCreate(savedInstanceState)
//
//        // read uri string
//        val cropUriString = intent.getStringExtra(EXTRA_CROP_URI)
//        val cropUri: Uri? = if (!cropUriString.isNullOrEmpty()) {
//            try {
//                Uri.parse(cropUriString)
//            } catch (e: Exception) {
//                null
//            }
//        } else null
//
//        // lazy init classifier in background (avoid blocking UI)
//        lifecycleScope.launch {
//            try {
//                classifier = Classifier(this@ResultActivity)
//            } catch (e: Throwable) {
//                e.printStackTrace()
//                Toast.makeText(this@ResultActivity, "Failed to load classifier: ${e.message}", Toast.LENGTH_LONG).show()
//            }
//        }
//
//        setContent {
//            MobileAppTheme {
//                Surface(
//                    modifier = Modifier.fillMaxSize(),
//                    color = MaterialTheme.colorScheme.background
//                ) {
//                    if (cropUri == null) {
//                        EmptyState()
//                    } else {
//                        ResultScreen(
//                            cropUri = cropUri,
//                            onBack = { finish() },
//                            onClassify = { uri, onResult ->
//                                // run classification in activity scope
//                                lifecycleScope.launch {
//                                    // ensure classifier loaded
//                                    if (classifier == null) {
//                                        try {
//                                            classifier = Classifier(this@ResultActivity)
//                                        } catch (e: Throwable) {
//                                            e.printStackTrace()
//                                            Toast.makeText(this@ResultActivity, "Classifier load error: ${e.message}", Toast.LENGTH_LONG).show()
//                                            onResult(null)
//                                            return@launch
//                                        }
//                                    }
//
//                                    // decode bitmap in IO
//                                    val bmp = withContext(Dispatchers.IO) {
//                                        decodeBitmapFromUri(uri)
//                                    }
//                                    if (bmp == null) {
//                                        Toast.makeText(this@ResultActivity, "Failed to decode image for classification", Toast.LENGTH_SHORT).show()
//                                        onResult(null)
//                                        return@launch
//                                    }
//
//                                    // predict (on Default)
//                                    val res = withContext(Dispatchers.Default) {
//                                        try {
//                                            classifier?.predict(bmp)
//                                        } catch (e: Throwable) {
//                                            e.printStackTrace()
//                                            null
//                                        }
//                                    }
//                                    onResult(res)
//                                }
//                            }
//                        )
//                    }
//                }
//            }
//        }
//    }
//
//    private fun decodeBitmapFromUri(uri: Uri, maxDim: Int = 512): Bitmap? {
//        try {
//            // decode bounds first
//            val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
//            var `in`: InputStream? = null
//            try {
//                `in` = contentResolver.openInputStream(uri)
//                BitmapFactory.decodeStream(`in`, null, options)
//            } finally {
//                `in`?.close()
//            }
//
//            val (w, h) = options.outWidth to options.outHeight
//            var inSample = 1
//            val largest = maxOf(w, h)
//            if (largest > maxDim) {
//                while (largest / inSample > maxDim) inSample *= 2
//            }
//
//            val opt2 = BitmapFactory.Options().apply { inSampleSize = inSample }
//            var `in2`: InputStream? = null
//            try {
//                `in2` = contentResolver.openInputStream(uri)
//                return BitmapFactory.decodeStream(`in2`, null, opt2)
//            } finally {
//                `in2`?.close()
//            }
//        } catch (e: Exception) {
//            e.printStackTrace()
//            return null
//        }
//    }
//}
//
//@Composable
//fun EmptyState() {
//    Box(
//        modifier = Modifier
//            .fillMaxSize()
//            .background(MaterialTheme.colorScheme.background),
//        contentAlignment = Alignment.Center
//    ) {
//        Text(text = "No cropped image provided", style = MaterialTheme.typography.bodyLarge)
//    }
//}
//
//@Composable
//fun ResultScreen(
//    cropUri: Uri,
//    onBack: () -> Unit,
//    onClassify: (uri: Uri, onResult: (com.example.mobileapp.inference.ClassifyResult?) -> Unit) -> Unit
//) {
//    var isClassifying by remember { mutableStateOf(false) }
//    var classLabel by remember { mutableStateOf<String?>(null) }
//    var classConf by remember { mutableStateOf<Float?>(null) }
//
//    val ctx = LocalContext.current
//
//    Column(
//        modifier = Modifier
//            .fillMaxSize()
//            .padding(16.dp),
//        horizontalAlignment = Alignment.CenterHorizontally
//    ) {
//        Text(text = "Detected sign", style = MaterialTheme.typography.headlineSmall)
//        Spacer(modifier = Modifier.height(12.dp))
//
//        Box(
//            modifier = Modifier
//                .fillMaxWidth()
//                .height(400.dp),
//            contentAlignment = Alignment.Center
//        ) {
//            Image(
//                painter = rememberAsyncImagePainter(cropUri),
//                contentDescription = "Cropped sign",
//                modifier = Modifier
//                    .fillMaxSize()
//                    .clip(RoundedCornerShape(8.dp))
//            )
//        }
//
//        Spacer(modifier = Modifier.height(16.dp))
//
//        if (isClassifying) {
//            Row(verticalAlignment = Alignment.CenterVertically) {
//                CircularProgressIndicator(modifier = Modifier.size(36.dp))
//                Spacer(modifier = Modifier.width(12.dp))
//                Text("Classifying...", style = MaterialTheme.typography.bodyMedium)
//            }
//        } else {
//            Row(
//                modifier = Modifier.fillMaxWidth(),
//                horizontalArrangement = Arrangement.SpaceEvenly
//            ) {
//                Button(onClick = { onBack() }, modifier = Modifier.weight(1f)) {
//                    Text("Back")
//                }
//                Spacer(modifier = Modifier.width(12.dp))
//                Button(onClick = {
//                    // start classification
//                    isClassifying = true
//                    classLabel = null
//                    classConf = null
//                    onClassify(cropUri) { result ->
//                        isClassifying = false
//                        if (result == null) {
//                            classLabel = "error"
//                            classConf = 0f
//                            Toast.makeText(ctx, "Classification failed", Toast.LENGTH_SHORT).show()
//                        } else {
//                            classLabel = result.label
//                            classConf = result.confidence
//                        }
//                    }
//                }, modifier = Modifier.weight(1f)) {
//                    Text("Classify")
//                }
//            }
//        }
//
//        Spacer(modifier = Modifier.height(12.dp))
//
//        if (classLabel != null) {
//            Text(text = "Label: ${classLabel}", style = MaterialTheme.typography.bodyLarge)
//            Text(text = String.format("Confidence: %.2f", classConf ?: 0f), style = MaterialTheme.typography.bodyMedium)
//        } else {
//            Text(
//                text = "Tip: when classifier is ready, press Classify to see the predicted sign.",
//                style = MaterialTheme.typography.bodySmall,
//                modifier = Modifier.padding(horizontal = 8.dp)
//            )
//        }
//    }
//}
