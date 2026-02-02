    package com.ml.quaterion.facenetdetection.model

class Models {

    companion object {

        val FACENET = ModelInfo(
            "FaceNet" ,
            "facenet.tflite" ,
            0.4f ,
            10f ,
            128 ,
            160
        )

        val FACENET_512 = ModelInfo(
            "FaceNet-512" ,
            "facenet_512.tflite" ,
            0.3f ,
            23.56f ,
            512 ,
            160
        )

        val FACENET_QUANTIZED = ModelInfo(
            "FaceNet Quantized" ,
            "facenet_int_quantized.tflite" ,
            0.4f ,
            10f ,
            128 ,
            160
        )

        val FACENET_512_QUANTIZED = ModelInfo(
            "FaceNet-512 Quantized" ,
            "facenet_512_int_quantized.tflite" ,
            0.3f ,
            23.56f ,
            512 ,
            160
        )
        
        val MOBILE_FACENET = ModelInfo(
            "MobileFaceNet", 
            "mobilefacenet.tflite", // The name of your file in assets
            112,                    // MobileFaceNet uses 112x112
            128                     // It spits out 128 numbers
        )
        
    }

}
