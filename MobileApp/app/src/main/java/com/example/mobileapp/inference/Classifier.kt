package com.example.mobileapp.inference

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.BufferedReader
import java.io.InputStreamReader
import kotlin.math.exp

data class ClassifyResult(val classId: Int, val label: String, val confidence: Float)

class Classifier(
    private val context: Context,
    private val modelAssetPath: String = "models/EnhancedLeNet5_best.ptl",
    private val labelsAssetPath: String = "models/labels.txt",
    private val inputSize: Int = 32 // ustaw na rozmiar, na którym model był trenowany
) {
    companion object {
        private const val TAG = "Classifier"
    }

    private val module: Module
    private val labels: List<String>

    init {
        // 1) load model file from assets (use your existing AssetUtils.assetFilePath)
        val modelPath = AssetUtils.assetFilePath(context, modelAssetPath)
        Log.i(TAG, "Loading model from: $modelPath")
        module = LiteModuleLoader.load(modelPath)
        labels = loadLabelsFromAssets(labelsAssetPath)
        if (labels.isEmpty()) Log.w(TAG, "Labels list is empty!")
        if (labels.size != 43) Log.i(TAG, "Loaded ${labels.size} labels (expected 43).")
    }

    private fun loadLabelsFromAssets(path: String): List<String> {
        val list = mutableListOf<String>()
        try {
            context.assets.open(path).use { stream ->
                BufferedReader(InputStreamReader(stream)).use { br ->
                    var line: String? = br.readLine()
                    while (line != null) {
                        val trimmed = line.trim()
                        if (trimmed.isNotEmpty()) list.add(trimmed)
                        line = br.readLine()
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load labels from assets/$path: ${e.message}")
        }
        return list
    }

    // Softmax helper
//    private fun softmax(logits: FloatArray): FloatArray {
//        var max = Float.NEGATIVE_INFINITY
//        for (v in logits) if (v > max) max = v
//        val exps = FloatArray(logits.size)
//        var sum = 0f
//        for (i in logits.indices) {
//            val e = exp((logits[i] - max).toDouble()).toFloat()
//            exps[i] = e
//            sum += e
//        }
//        for (i in exps.indices) exps[i] = exps[i] / sum
//        return exps
//    }

    // safe softmax
    private fun softmax(logits: FloatArray): FloatArray {
        val max = logits.maxOrNull() ?: 0f
        val exps = FloatArray(logits.size)
        var sum = 0f
        for (i in logits.indices) {
            val e = kotlin.math.exp((logits[i] - max).toDouble()).toFloat()
            exps[i] = e
            sum += e
        }
        for (i in exps.indices) exps[i] = exps[i] / sum
        return exps
    }

    // Główna funkcja inferencji: przyjmuje już wycięty znak (croppedBitmap)
    fun predict(croppedBitmap: Bitmap): ClassifyResult? {
        // 1) Resize to model input size (nie zachowujemy aspect, bo model trenowany na fixed input)
        val inputBmp = Bitmap.createScaledBitmap(croppedBitmap, inputSize, inputSize, true)

        // 2) Preprocessing: dokładnie ta sama normalizacja co w treningu: (x/255 - 0.5)/0.5 -> => mean=0.5,std=0.5
        val mean = floatArrayOf(0.5f, 0.5f, 0.5f)
        val std = floatArrayOf(0.5f, 0.5f, 0.5f)

        // TensorImageUtils.bitmapToFloat32Tensor zwraca Tensor o kształcie (1,3,H,W) zgodny z PyTorch
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(inputBmp, mean, std)
        val inputArr = inputTensor.dataAsFloatArray
        val inMax = inputArr.maxOrNull()
        val inMin = inputArr.minOrNull()
        val inMean = inputArr.average()
        Log.i("Debug", "input min=$inMin max=$inMax mean=$inMean")


        // 3) Forward
        val outputTensor: Tensor = module.forward(IValue.from(inputTensor)).toTensor()
        val out = outputTensor.dataAsFloatArray
        val sum = out.sum()
        Log.i("Debug", "out sum = $sum, max = ${out.maxOrNull()}, min = ${out.minOrNull()}")
        val max = out.maxOrNull() ?: 0f
        val avg = out.average().toFloat()
        Log.i("Debug", "max=$max avg=$avg")


        // 4) Odczytaj wynik i skonwertuj do probabilities
        val outShape = outputTensor.shape() // np. [1,43] lub [43] etc.
        val outArray = outputTensor.dataAsFloatArray
        val probs: FloatArray
        if (outShape.size == 2 && outShape[0] == 1L && outShape[1].toInt() == outArray.size) {
            // shape (1, N)
            probs = softmax(outArray)
        } else if (outShape.size == 1 && outShape[0] == outArray.size.toLong()) {
            // shape (N)
            probs = softmax(outArray)
        } else {
            // niespodziewany kształt: spróbuj traktować jako logits i softmax
            probs = softmax(outArray)
            Log.w(TAG, "Unexpected output shape: ${outShape.joinToString(",")}, treated as logits length=${outArray.size}")
        }

        data class PairIdx(val idx: Int, val prob: Float)
        val top = probs.mapIndexed { i, p -> PairIdx(i, p) }
            .sortedByDescending { it.prob }
            .take(3)

        Log.i("ClassifierDebug", "Top3 probs: ${top.map { "${it.idx}:${"%.4f".format(it.prob)}" }}")

        // 5) argmax
        var maxIdx = 0
        var maxVal = probs[0]
        for (i in 1 until probs.size) {
            if (probs[i] > maxVal) {
                maxVal = probs[i]
                maxIdx = i
            }
        }

        val label = if (maxIdx < labels.size) labels[maxIdx] else "class_$maxIdx"
        return ClassifyResult(maxIdx, label, maxVal)
    }
}
