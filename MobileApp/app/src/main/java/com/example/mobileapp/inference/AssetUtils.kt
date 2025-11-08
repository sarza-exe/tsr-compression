package com.example.mobileapp.inference

import android.content.Context
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

object AssetUtils {

    /**
     * Kopiuje asset z katalogu `assets` do internal files dir aplikacji i zwraca absolutną ścieżkę do skopiowanego pliku.
     *
     * @param context Context aplikacji / activity
     * @param assetName Ścieżka do assetu w katalogu assets, np. "models/gtsdb_yolo_416.ptl"
     * @return absolutna ścieżka do skopiowanego pliku w internal storage
     * @throws IOException jeśli kopiowanie się nie powiedzie
     */
    @Throws(IOException::class)
    fun assetFilePath(context: Context, assetName: String): String {
        val outFile = File(context.filesDir, assetName)

        // jeśli plik już istnieje i ma rozmiar > 0, zwróć jego ścieżkę
        if (outFile.exists() && outFile.length() > 0) {
            return outFile.absolutePath
        }

        // upewnij się, że katalogi nadrzędne istnieją (jeśli assetName zawiera podkatalogi)
        outFile.parentFile?.let { parent ->
            if (!parent.exists()) {
                parent.mkdirs()
            }
        }

        context.assets.open(assetName).use { input ->
            FileOutputStream(outFile).use { output ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (input.read(buffer).also { read = it } != -1) {
                    output.write(buffer, 0, read)
                }
                output.flush()
            }
        }

        if (!outFile.exists() || outFile.length() == 0L) {
            throw IOException("Failed to copy asset file: $assetName")
        }

        return outFile.absolutePath
    }
}
