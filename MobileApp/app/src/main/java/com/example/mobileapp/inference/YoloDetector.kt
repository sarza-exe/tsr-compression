package com.example.mobileapp.inference

import android.content.Context
import android.graphics.*
import android.util.Log
import com.example.mobileapp.inference.AssetUtils.assetFilePath
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.Console
import kotlin.math.max
import kotlin.math.min

data class Detection(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val score: Float,
    val label: Int
)

class YoloDetector(context: Context) {

    companion object {
        private const val TAG = "YoloDetector"
        private const val MODEL_ASSET_PATH = "models/gtsdb_yolo_416_3.ptl"
        private const val INPUT_SIZE = 416                       // model input H/W
        private const val PAD_COLOR = 114                        // letterbox pad RGB value (114,114,114)
        private const val DEFAULT_CONF_THRESHOLD = 0.25f
        private const val DEFAULT_NMS_IOU = 0.45f
    }

    private val module: Module

    init {
        // Use applicationContext to avoid leaking Activity
        val appCtx = context.applicationContext

        try {
            val modelPath = assetFilePath(appCtx, MODEL_ASSET_PATH)
            module = LiteModuleLoader.load(modelPath)
            Log.i(TAG, "Loaded model from: $modelPath")
        } catch (e: Exception) {
            // catch IOException, UnsatisfiedLinkError, IllegalArgumentException etc.
            Log.e(TAG, "Failed to load model: ${e.message}", e)
            throw RuntimeException("YoloDetector initialization failed: ${e.message}", e)
        }
    }

    /**
     * Główna funkcja detekcji.
     * @param bitmap wejściowy obraz (oryginalne rozmiary)
     * @param confThreshold filtr pewności (score)
     * @param nmsIouThreshold IoU threshold dla NMS
     * @return lista detekcji z koordynatami w pikselach oryginalnego obrazu
     */
    fun detect(
        bitmap: Bitmap,
        confThreshold: Float = DEFAULT_CONF_THRESHOLD,
        nmsIouThreshold: Float = DEFAULT_NMS_IOU
    ): List<Detection> {
        // 1) letterbox -> przygotuj input bitmap (416x416) oraz parametry mapowania
        val (paddedBitmap, scale, padX, padY) = letterbox(bitmap, INPUT_SIZE, INPUT_SIZE, PAD_COLOR)

        // 2) przygotuj tensor (1,3,416,416) RGB, normalized [0..1]
        val inputTensor = bitmapToFloat32Tensor(paddedBitmap)

        // 3) forward
        val outputTensor: Tensor = module.forward(IValue.from(inputTensor)).toTensor()

        // after val outputTensor: Tensor = module.forward(IValue.from(inputTensor)).toTensor()
        val shape = outputTensor.shape() // LongArray
        Log.i("YoloDebug", "outputTensor.shape = ${shape.joinToString(",")}")

// If it's a 2D/3D tensor, try to print first row(s) nicely
        try {
            val outArray = outputTensor.dataAsFloatArray
            Log.i("YoloDebug", "outArray.len=${outArray.size}")
            // If we can infer dims: attempt rows = outArray.size / shape.last()
            val lastDim = if (shape.isNotEmpty()) shape.last().toInt() else -1
            if (lastDim > 0) {
                val rows = outArray.size / lastDim
                Log.i("YoloDebug", "Inferred rows=$rows, cols=$lastDim")
                val previewRows = minOf(rows, 5)
                for (r in 0 until previewRows) {
                    val sb = StringBuilder()
                    for (c in 0 until lastDim) {
                        val v = outArray[r * lastDim + c]
                        sb.append(String.format("%.4f ", v))
                    }
                    Log.i("YoloDebug", "row[$r]: $sb")
                }
            } else {
                // fallback preview
                val previewCount = minOf(outArray.size, 60)
                val sb = StringBuilder()
                for (k in 0 until previewCount) sb.append(String.format("%.4f ", outArray[k]))
                Log.i("YoloDebug", "outArray preview: $sb")
            }
        } catch (e: Exception) {
            Log.w("YoloDebug", "Could not dump output tensor: ${e.message}")
        }

        val outArray = outputTensor.dataAsFloatArray

        Log.i("YoloDebug", "outArray.len=${outArray.size}")
        // print first 30 values (or all if small)
        val previewCount = minOf(outArray.size, 30)
        val sb = StringBuilder()
        for (k in 0 until previewCount) {
            sb.append(String.format("%.4f ", outArray[k]))
        }
        Log.i("FEFREOGHREOHGOREGHROEGYoloDebug", "outArray preview: $sb")


        // 4) parse output (flattened N x 6)
        val detections = mutableListOf<Detection>()
        var i = 0
        while (i + 5 < outArray.size) {
            val x1n = outArray[i]
            val y1n = outArray[i + 1]
            val x2n = outArray[i + 2]
            val y2n = outArray[i + 3]
            val score = outArray[i + 4]
            val classF = outArray[i + 5]
            i += 6

            if (score < confThreshold) continue

            // zamiana z normalizowanego (0..1) -> współrzędne w inputie (416x416)
            val x1In = x1n * INPUT_SIZE
            val y1In = y1n * INPUT_SIZE
            val x2In = x2n * INPUT_SIZE
            val y2In = y2n * INPUT_SIZE

            // odwrócenie letterbox: (in_coords - pad) / scale => coords w oryginalnym obrazie
            val x1Orig = ((x1In - padX) / scale).coerceIn(0f, bitmap.width.toFloat())
            val y1Orig = ((y1In - padY) / scale).coerceIn(0f, bitmap.height.toFloat())
            val x2Orig = ((x2In - padX) / scale).coerceIn(0f, bitmap.width.toFloat())
            val y2Orig = ((y2In - padY) / scale).coerceIn(0f, bitmap.height.toFloat())

            val label = classF.toInt()
            detections.add(Detection(x1Orig, y1Orig, x2Orig, y2Orig, score, label))
        }

        // 5) NMS (w pixelach oryginalnych)
        val final = nms(detections, nmsIouThreshold)
        return final
    }

    // Helpers

    /**
     * Letterbox: scale with aspect ratio and pad to targetW x targetH
     * Returns: paddedBitmap, scale, padX, padY
     * padX/padY are floats in pixels relative to target image (i.e., 0..INPUT_SIZE)
     */
    private fun letterbox(src: Bitmap, targetW: Int, targetH: Int, padColorValue: Int): QuadrupleBitmapFloat {
        val origW = src.width.toFloat()
        val origH = src.height.toFloat()

        val scale = min(targetW / origW, targetH / origH)
        val newW = (origW * scale).toInt()
        val newH = (origH * scale).toInt()

        val padX = (targetW - newW) / 2f
        val padY = (targetH - newH) / 2f

        // create target bitmap filled with pad color
        val padded = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(padded)
        val padColor = Color.rgb(padColorValue, padColorValue, padColorValue)
        canvas.drawColor(padColor)

        // draw scaled src onto canvas at (padX, padY)
        val srcRect = Rect(0, 0, src.width, src.height)
        val dstRect = RectF(padX, padY, padX + newW, padY + newH)
        val paint = Paint().apply { isFilterBitmap = true }
        canvas.drawBitmap(src, srcRect, dstRect, paint)

        return QuadrupleBitmapFloat(padded, scale, padX, padY)
    }

    /**
     * Konwersja padded bitmap do Float32 tensoru (1,3,H,W) RGB normalized [0..1].
     * Order: channel-first (CHW)
     */
    private fun bitmapToFloat32Tensor(bmp: Bitmap): Tensor {
        val w = bmp.width
        val h = bmp.height
        val floatBuffer = FloatArray(1 * 3 * h * w) // CHW
        val pixels = IntArray(w * h)
        bmp.getPixels(pixels, 0, w, 0, 0, w, h) // ARGB ints

        // fill array: channel R then G then B
        val planeSize = w * h
        for (y in 0 until h) {
            val rowOffset = y * w
            for (x in 0 until w) {
                val px = pixels[rowOffset + x]
                val r = ((px shr 16) and 0xFF).toFloat() / 255.0f
                val g = ((px shr 8) and 0xFF).toFloat() / 255.0f
                val b = ((px) and 0xFF).toFloat() / 255.0f

                val idxR = 0 * planeSize + rowOffset + x
                val idxG = 1 * planeSize + rowOffset + x
                val idxB = 2 * planeSize + rowOffset + x

                floatBuffer[idxR] = r
                floatBuffer[idxG] = g
                floatBuffer[idxB] = b
            }
        }

        return Tensor.fromBlob(floatBuffer, longArrayOf(1, 3, h.toLong(), w.toLong()))
    }

    /**
     * Simple NMS implementation (IoU based). Input coords are in same coordinate system (pixels original image).
     */
    private fun nms(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        val dets = detections.sortedByDescending { it.score }.toMutableList()
        val out = mutableListOf<Detection>()
        while (dets.isNotEmpty()) {
            val best = dets.removeAt(0)
            out.add(best)
            val it = dets.iterator()
            while (it.hasNext()) {
                val other = it.next()
                if (iou(best, other) > iouThreshold) {
                    it.remove()
                }
            }
        }
        return out
    }

    private fun iou(a: Detection, b: Detection): Float {
        val interLeft = max(a.x1, b.x1)
        val interTop = max(a.y1, b.y1)
        val interRight = min(a.x2, b.x2)
        val interBottom = min(a.y2, b.y2)

        val interW = max(0f, interRight - interLeft)
        val interH = max(0f, interBottom - interTop)
        val interArea = interW * interH
        val areaA = max(0f, (a.x2 - a.x1)) * max(0f, (a.y2 - a.y1))
        val areaB = max(0f, (b.x2 - b.x1)) * max(0f, (b.y2 - b.y1))
        val union = areaA + areaB - interArea
        return if (union <= 0f) 0f else interArea / union
    }

    // small helper data holder (no other language features required)
    private data class QuadrupleBitmapFloat(val bitmap: Bitmap, val scale: Float, val padX: Float, val padY: Float)
}
