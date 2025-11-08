package com.example.mobileapp.ui.main

import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import coil.compose.rememberAsyncImagePainter
import com.example.mobileapp.ui.theme.MobileAppTheme

class ResultActivity : ComponentActivity() {

    companion object {
        const val EXTRA_CROP_URI = "crop_uri"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Odczytujemy URI z intent (przekazane jako string)
        val cropUriString = intent.getStringExtra(EXTRA_CROP_URI)
        val cropUri: Uri? = if (!cropUriString.isNullOrEmpty()) {
            try {
                Uri.parse(cropUriString)
            } catch (e: Exception) {
                null
            }
        } else null

        setContent {
            MobileAppTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    if (cropUri == null) {
                        EmptyState()
                    } else {
                        ResultScreen(
                            cropUri = cropUri,
                            onBack = { finish() },
                            onClassify = {
                                // Placeholder: klasyfikacja nie jest jeszcze zaimplementowana.
                                Toast.makeText(this, "Classifier not implemented yet", Toast.LENGTH_SHORT).show()
                            }
                        )
                    }
                }
            }
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

@Composable
fun ResultScreen(
    cropUri: Uri,
    onBack: () -> Unit,
    onClassify: () -> Unit
) {
    val context = LocalContext.current

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(text = "Detected sign", style = MaterialTheme.typography.headlineSmall)
        Spacer(modifier = Modifier.height(12.dp))

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(400.dp),
            contentAlignment = Alignment.Center
        ) {
            // Coil: ładuje URI (obsługuje file:// content:// itp.)
            Image(
                painter = rememberAsyncImagePainter(cropUri),
                contentDescription = "Cropped sign",
                modifier = Modifier
                    .fillMaxSize()
                    .clip(RoundedCornerShape(8.dp))
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            Button(onClick = { onBack() }, modifier = Modifier.weight(1f)) {
                Text("Back")
            }
            Spacer(modifier = Modifier.width(12.dp))
            Button(onClick = {
                // Tu kiedyś uruchomimy model TSR (classifier)
                onClassify()
            }, modifier = Modifier.weight(1f)) {
                Text("Classify")
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        Text(
            text = "Tip: when classifier is ready, press Classify to see the predicted sign.",
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.padding(horizontal = 8.dp)
        )
    }
}
