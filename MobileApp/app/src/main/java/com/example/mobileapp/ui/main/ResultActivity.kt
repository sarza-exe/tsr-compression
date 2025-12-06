package com.example.mobileapp.ui.main

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.clickable
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import coil.compose.AsyncImage
import com.example.mobileapp.inference.Classifier
import com.example.mobileapp.inference.ClassifyResult
import com.example.mobileapp.ui.theme.MobileAppTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.InputStream

class ResultActivity : ComponentActivity() {

    companion object {
        const val EXTRA_CROP_URI = "crop_uri"
    }

    // Cache classifier instances per model file (to avoid repeated reload)
    private val classifierCache = mutableMapOf<String, Classifier?>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // read uri string from intent
        val cropUriString = intent.getStringExtra(EXTRA_CROP_URI)
        val cropUri: Uri? = cropUriString?.let { Uri.parse(it) }

        // read available model files in assets/models
        val modelFiles = try {
            // list names under assets/models (returns file names)
            assets.list("models")
                ?.filter { it.endsWith(".ptl") && it != "gtsdb_yolo_416_3.ptl" }
                ?.sorted()?: emptyList()
        } catch (e: Exception) {
            emptyList()
        }

        setContent {
            MobileAppTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    if (cropUri == null) {
                        EmptyState()
                    } else {
                        // Provide modelFiles to the composable
                        ResultScreenWithModels(
                            cropUri = cropUri,
                            modelFiles = modelFiles,
                            onBack = { finish() },
                            // onClassify: the activity will perform classification when composable requests it
                            onClassify = { selectedModelFile, uri, onResult ->
                                // run classification in activity lifecycle scope
                                lifecycleScope.launch {
                                    // decode image
                                    val bmp = withContext(Dispatchers.IO) {
                                        decodeBitmapFromUri(uri)
                                    }
                                    if (bmp == null) {
                                        Toast.makeText(this@ResultActivity, "Failed to decode image to classify", Toast.LENGTH_SHORT).show()
                                        onResult(null)
                                        return@launch
                                    }

                                    // determine labels asset for this model (try <base>_labels.txt then fallback to models/labels.txt)
                                    val base = selectedModelFile.removeSuffix(".ptl")
                                    val candidate1 = "${base}_labels.txt"
                                    val labelsAsset = when {
                                        assetExistsInModels(candidate1) -> "models/$candidate1"
                                        assetExistsInModels("labels.txt") -> "models/labels.txt"
                                        else -> "models/labels.txt" // fallback even if not present
                                    }

                                    val modelAssetPath = "models/$selectedModelFile"

                                    // get or create classifier for this model
                                    val classifier = classifierCache.getOrPut(modelAssetPath) {
                                        try {
                                            Classifier(this@ResultActivity, modelAssetPath = modelAssetPath, labelsAssetPath = labelsAsset)
                                        } catch (e: Throwable) {
                                            e.printStackTrace()
                                            null
                                        }
                                    }

                                    if (classifier == null) {
                                        Toast.makeText(this@ResultActivity, "Failed to load classifier for $selectedModelFile", Toast.LENGTH_LONG).show()
                                        onResult(null)
                                        return@launch
                                    }

                                    // predict (on Default)
                                    val res = withContext(Dispatchers.Default) {
                                        try {
                                            classifier?.predict(bmp)
                                        } catch (e: Throwable) {
                                            e.printStackTrace()
                                            null
                                        }
                                    }
                                    onResult(res)
                                }
                            }
                        )
                    }
                }
            }
        }
    }

    private fun assetExistsInModels(name: String): Boolean {
        return try {
            assets.list("models")?.contains(name) == true
        } catch (e: Exception) {
            false
        }
    }

    private fun decodeBitmapFromUri(uri: Uri, maxDim: Int = 512): Bitmap? {
        try {
            // decode bounds first to set inSampleSize
            val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            var `in`: InputStream? = null
            try {
                `in` = contentResolver.openInputStream(uri)
                BitmapFactory.decodeStream(`in`, null, options)
            } finally {
                `in`?.close()
            }

            val (w, h) = options.outWidth to options.outHeight
            var inSample = 1
            val largest = maxOf(w, h)
            if (largest > maxDim) {
                while (largest / inSample > maxDim) inSample *= 2
            }

            val opt2 = BitmapFactory.Options().apply { inSampleSize = inSample }
            var `in2`: InputStream? = null
            try {
                `in2` = contentResolver.openInputStream(uri)
                return BitmapFactory.decodeStream(`in2`, null, opt2)
            } finally {
                `in2`?.close()
            }
        } catch (e: Exception) {
            e.printStackTrace()
            return null
        }
    }
}

@Composable
fun EmptyState() {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
        contentAlignment = Alignment.Center
    ) {
        Text(text = "No cropped image provided", style = MaterialTheme.typography.bodyLarge)
    }
}

/* Compose UI below: ResultScreenWithModels shows image, Classify button and model-selection row.
   It guarantees exactly one model selected (first one by default). */

@Composable
fun ResultScreenWithModels(
    cropUri: Uri,
    modelFiles: List<String>,
    onBack: () -> Unit,
    onClassify: (selectedModelFile: String, uri: Uri, onResult: (com.example.mobileapp.inference.ClassifyResult?) -> Unit) -> Unit
) {
    var isClassifying by remember { mutableStateOf(false) }
    var classLabel by remember { mutableStateOf<String?>(null) }
    var classConf by remember { mutableStateOf<Float?>(null) }
    var top3 by remember { mutableStateOf<List<Pair<Int, Float>>?>(null) }

    // ensure at least one selected index (0 if list not empty)
    var selectedIdx by remember { mutableStateOf(if (modelFiles.isNotEmpty()) 0 else -1) }

    val ctx = LocalContext.current
    val scrollState = rememberScrollState() // 1. Stan scrollowania

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(12.dp)
            .verticalScroll(scrollState),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(modifier = Modifier.height(24.dp))
        Text(text = "Detected sign", style = MaterialTheme.typography.headlineSmall)

        Spacer(modifier = Modifier.height(16.dp))

        Box(
            modifier = Modifier
                .fillMaxWidth(0.9f)
                .height(300.dp),
            contentAlignment = Alignment.Center
        ) {
            AsyncImage(
                model = cropUri,
                contentDescription = "Cropped sign",
                modifier = Modifier
                    .fillMaxSize()
                    .clip(RoundedCornerShape(8.dp))
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        // controls
        if (isClassifying) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(modifier = Modifier.size(36.dp))
                Spacer(modifier = Modifier.width(12.dp))
                Text("Classifying...", style = MaterialTheme.typography.bodyMedium)
            }
        } else {
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                Button(onClick = { onBack() }, modifier = Modifier.weight(1f)) {
                    Text("Back")
                }
                Spacer(modifier = Modifier.width(12.dp))
                Button(onClick = {
                    // ensure a model is selected
                    if (selectedIdx < 0 || selectedIdx >= modelFiles.size) {
                        Toast.makeText(ctx, "No model selected", Toast.LENGTH_SHORT).show()
                        return@Button
                    }
                    val selectedModel = modelFiles[selectedIdx]
                    // start classification via provided callback
                    isClassifying = true
                    classLabel = null
                    classConf = null
                    top3 = null
                    onClassify(selectedModel, cropUri) { result ->
                        isClassifying = false
                        if (result == null) {
                            classLabel = "error"
                            classConf = 0f
                            Toast.makeText(ctx, "Classification failed", Toast.LENGTH_SHORT).show()
                        } else {
                            classLabel = result.label
                            classConf = result.confidence
                        }
                    }
                }, modifier = Modifier.weight(1f)) {
                    Text("Classify")
                }
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // show result
        if (classLabel != null) {
            Text(text = "Label: ${classLabel}", style = MaterialTheme.typography.bodyLarge)
            Text(text = String.format("Confidence: %.2f", classConf ?: 0f), style = MaterialTheme.typography.bodyMedium)
        } else {
            Text(
                text = "Tip: pick model and press Classify.",
                style = MaterialTheme.typography.bodySmall
            )
        }

        Spacer(modifier = Modifier.height(12.dp))

        // model selector (horizontal row of chips / buttons) - always one selected
        Text(text = "Models:", style = MaterialTheme.typography.bodyLarge)
        Spacer(modifier = Modifier.height(8.dp))

        if (modelFiles.isEmpty()) {
            Text(text = "No models found in assets/models/", style = MaterialTheme.typography.bodyMedium)
        } else {
            LazyRow(modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                itemsIndexed(modelFiles) { index, modelFile ->
                    val isSelected = index == selectedIdx
                    val displayName = modelFile.removeSuffix(".ptl")
                    Surface(
                        tonalElevation = if (isSelected) 8.dp else 0.dp,
                        shape = RoundedCornerShape(16.dp),
                        color = if (isSelected) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surface,
                        modifier = Modifier
                            .padding(4.dp)
                            .clickable {
                                // select this model (cannot unselect; exactly one selected)
                                selectedIdx = index
                            }
                    ) {
                        Text(
                            text = displayName,
                            modifier = Modifier
                                .padding(horizontal = 12.dp, vertical = 10.dp),
                            style = MaterialTheme.typography.bodyMedium,
                            color = if (isSelected) MaterialTheme.colorScheme.onPrimaryContainer else MaterialTheme.colorScheme.onSurface
                        )
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(8.dp))
    }
}