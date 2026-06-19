package com.safellmkit.ml.tokenizer

interface Tokenizer {
    fun tokenize(text: String, maxLen: Int = 256): TokenizedInput
}
