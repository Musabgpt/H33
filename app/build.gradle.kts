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
        versionCode = 7
        versionName = "0.7.0-deepseek-gguf"

        ndk {
            abiFilters += listOf("arm64-v8a")
        }
        externalNativeBuild { cmake { cppFlags += listOf("-O3", "-fvisibility=hidden") } }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    androidResources {
        noCompress += listOf("gguf")
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    packaging { resources.excludes += setOf("META-INF/**", "kotlin/**") }
}

dependencies {
    implementation("androidx.appcompat:appcompat:1.7.1")
    implementation("com.google.android.material:material:1.13.0")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20250517")
}
