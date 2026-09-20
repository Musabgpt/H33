plugins {
    id("com.android.application")
}

android {
    namespace = "com.musab.h33bench"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.musab.h33bench"
        minSdk = 23
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
}

dependencies {
    implementation("com.google.mlkit:translate:17.0.3")
    androidTestImplementation("androidx.test.ext:junit:1.3.0")
    androidTestImplementation("androidx.test:runner:1.7.0")
}
