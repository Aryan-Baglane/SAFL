plugins {
    kotlin("multiplatform")
    kotlin("plugin.serialization")
    id("com.android.library")
    id("maven-publish")
}

kotlin {
    androidTarget {
        compilations.all {
            kotlinOptions {
                jvmTarget = "1.8"
            }
        }
    }
    jvm()

    // Enable after Xcode is configured
    iosX64()
    iosArm64()
    iosSimulatorArm64()

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")
            }
        }

        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
            }
        }

        val jvmMain by getting {
            dependencies {
                implementation("com.microsoft.onnxruntime:onnxruntime:1.18.0")
            }
        }

        // Android will use onnxruntime-android (added later when we add android target)
        // iOS will use onnxruntime framework / CocoaPods
        val androidMain by getting {
            dependencies {
                implementation("com.microsoft.onnxruntime:onnxruntime-android:1.18.0")
            }
        }
    }
}

android {
    namespace = "com.safellmkit.ml"
    compileSdk = 34
    defaultConfig {
        minSdk = 24
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
}
