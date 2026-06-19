package com.safellmkit.ml.tokenizer

interface TextTokenizer {
    fun encode(text: String, maxLength: Int = 256): TokenizedInput
}

data class TokenizedInput(
    val inputIds: LongArray,
    val attentionMask: LongArray
)

class BasicWordTokenizer(
    vocabLines: List<String>,
    private val doLowerCase: Boolean = true
) : TextTokenizer {

    private val vocab: Map<String, Int> = vocabLines.mapIndexed { index, token -> token.trim() to index }.toMap()

    override fun encode(text: String, maxLength: Int): TokenizedInput {
        val tokens = tokenize(text)
        val ids = ArrayList<Long>(maxLength)

        val clsId = vocab["[CLS]"] ?: 101
        val sepId = vocab["[SEP]"] ?: 102
        val unkId = vocab["[UNK]"] ?: 100
        val padId = vocab["[PAD]"] ?: 0

        ids.add(clsId.toLong())

        for (token in tokens) {
            val subTokens = wordpieceTokenize(token)
            for (subToken in subTokens) {
                if (ids.size >= maxLength - 1) break
                val id = vocab[subToken] ?: unkId
                ids.add(id.toLong())
            }
        }

        if (ids.size < maxLength) {
            ids.add(sepId.toLong())
        }

        while (ids.size < maxLength) {
            ids.add(padId.toLong())
        }

        val attention = LongArray(maxLength) { idx ->
            if (idx < ids.indexOfLast { it != padId.toLong() } + 1) 1L else 0L
        }

        return TokenizedInput(ids.toLongArray(), attention)
    }

    private fun wordpieceTokenize(word: String): List<String> {
        if (vocab.containsKey(word)) {
            return listOf(word)
        }
        val subTokens = mutableListOf<String>()
        var start = 0
        while (start < word.length) {
            var end = word.length
            var curSubToken: String? = null
            while (start < end) {
                var substr = word.substring(start, end)
                if (start > 0) {
                    substr = "##$substr"
                }
                if (vocab.containsKey(substr)) {
                    curSubToken = substr
                    break
                }
                end--
            }
            if (curSubToken == null) {
                return listOf("[UNK]")
            }
            subTokens.add(curSubToken)
            start = end
        }
        return subTokens
    }

    private fun tokenize(text: String): List<String> {
        val normalized = if (doLowerCase) text.lowercase() else text
        return normalized
            .replace(Regex("([.,!?;:()\\[\\]{}\"'])"), " $1 ")
            .split(Regex("\\s+"))
            .filter { it.isNotBlank() }
    }
}
