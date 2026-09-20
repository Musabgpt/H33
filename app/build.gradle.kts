plugins {
    id("com.android.application")
}

android {
    namespace = "com.musab.aragpt2"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.musab.aragpt2"
        minSdk = 28
        targetSdk = 36
        versionCode = 5
        versionName = "0.5.0-deepseek-coder"

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
        noCompress += listOf("onnx", "data", "json", "txt", "model")
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
}

dependencies {
    implementation("androidx.appcompat:appcompat:1.7.1")
    implementation("com.google.android.material:material:1.13.0")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.23.0")
    implementation(files("libs/onnxruntime-genai.aar"))
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20250517")
}
