package com.safellmkit.core.engine

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.alloc
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import platform.posix.gettimeofday
import platform.posix.timeval

actual fun loadCentroidsDiagnostics(): String {
    return "Centroids diagnostics not supported on iOS"
}

@OptIn(ExperimentalForeignApi::class)
actual fun currentTimeMs(): Long = memScoped {
    val tv = alloc<timeval>()
    gettimeofday(tv.ptr, null)
    tv.tv_sec * 1000L + tv.tv_usec / 1000L
}
