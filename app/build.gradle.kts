plugins {
    id("com.android.application")
}

android {
    namespace = "com.musab.aragpt2"
    compileSdk = 37

    defaultConfig {
        applicationId = "com.musab.aragpt2"
        minSdk = 28
        targetSdk = 37
        versionCode = 2
        versionName = "0.2.0"

        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    androidResources {
        noCompress += listOf("onnx", "json", "txt")
    }
}

dependencies {
    implementation("androidx.appcompat:appcompat:1.7.1")
    implementation("com.google.android.material:material:1.13.0")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.30.0")
}
