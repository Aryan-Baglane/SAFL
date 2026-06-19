plugins {
    kotlin("multiplatform")
    kotlin("plugin.serialization")
    id("maven-publish")
}
repositories{
    mavenCentral()
}

kotlin {
    jvm()

    iosX64()
    iosArm64()
    iosSimulatorArm64()

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.8.1")
                implementation("io.ktor:ktor-client-core:2.3.9")
                implementation("io.ktor:ktor-client-content-negotiation:2.3.9")
                implementation("io.ktor:ktor-serialization-kotlinx-json:2.3.9")
                implementation("io.ktor:ktor-network:2.3.9")
                implementation(project(":safellmkit-ml"))
            }
        }

        val jvmMain by getting {
            dependencies {
                implementation("io.ktor:ktor-client-cio:2.3.9")
            }
        }

        val iosMain by creating {
            dependsOn(commonMain)
            dependencies {
                implementation("io.ktor:ktor-client-darwin:2.3.9")
            }
        }

        listOf(
            iosX64().compilations["main"].defaultSourceSet,
            iosArm64().compilations["main"].defaultSourceSet,
            iosSimulatorArm64().compilations["main"].defaultSourceSet
        ).forEach { it.dependsOn(iosMain) }

        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.8.1")
            }
        }

        val jvmTest by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.8.1")
            }
        }
    }
}
