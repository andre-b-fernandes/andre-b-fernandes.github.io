---
layout: home
title:  "Byte Pair Encoding Tokenizer in C++"
date:  2026-04-10
permalink: /byte-pair-encoding-cpp/
categories: cpp nlp tokenization llm
image: byte-pair-encoding-cpp.png
---

# Introduction

In this post I am going to discuss my C++ implementation of the Byte Pair Encoding (BPE) tokenizer, which is part of a personal project I have been working on to build Large Language Model components from scratch in C++.

Tokenization is the very first step in any NLP pipeline — before a model can process text, the text must be converted into a sequence of integers that the model can work with. BPE is one of the most widely used tokenization algorithms today and is the basis for the tokenizers used in GPT-2, GPT-3, RoBERTa, and many other models.

BPE was originally proposed as a [data compression algorithm](https://arxiv.org/abs/1508.07909) by Philip Gage in 1994. The core idea is simple: iteratively replace the most frequent pair of bytes (or characters) in a sequence with a new symbol. Sennrich et al. (2016) adapted this technique for NLP tokenization, and it has since become ubiquitous. The [Hugging Face NLP course](https://huggingface.co/learn/nlp-course/chapter6/5) provides an excellent overview of how BPE works in the context of modern language models.

In this post I am going to walk through the four main stages of the BPE algorithm — pre-tokenization, training, encoding, and decoding — and show the C++ code I wrote for each one. I will also discuss key design decisions, complexity analysis, and how BPE compares to other tokenization approaches.

# The Four Stages of BPE

The BPE tokenizer works in four stages:

1. **Pre-tokenization** — split raw text into base units (words/characters)
2. **Training** — learn merge rules from a training corpus
3. **Encoding** — apply learned merge rules to convert text to token IDs
4. **Decoding** — convert token IDs back to text

Let me walk through each stage with the actual implementation.

## Stage 1: Pre-tokenization

Before BPE can operate, raw text must be broken down into an initial set of units. I use a GPT-style regex pattern that handles English contractions (`'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`), words, numbers, and punctuation:

```cpp
std::vector<std::string> BPETokenizer::pre_tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::regex pattern(
        "'s|'t|'re|'ve|'m|'ll|'d|"
        "[^\\r\\n\\w]?[a-zA-Z]+|"
        "[0-9]{1,3}|"
        " ?[^\\s\\w]+[\\r\\n]*|"
        "\\s*[\\r\\n]+|"
        "\\s+(?!\\S)|"
        "\\s+",
        std::regex::icase
    );
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), pattern);
    auto words_end = std::sregex_iterator();
    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        std::string token = match.str();
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}
```

This pattern closely mirrors the one used in GPT-2's tokenizer. By handling contractions as separate tokens (`don` and `'t` rather than `don't`), we avoid artificially linking word stems to suffixes — which would lead to poor generalization during training.

Once we have the list of pre-tokenized words, each word is split into individual characters. I also append an end-of-word marker `</w>` to the last character of each word:

```cpp
std::vector<std::string> BPETokenizer::word_to_chars(const std::string& word) {
    std::vector<std::string> chars;
    for (size_t i = 0; i < word.length(); ++i) {
        if (i == word.length() - 1) {
            chars.push_back(std::string(1, word[i]) + END_OF_WORD);
        } else {
            chars.push_back(std::string(1, word[i]));
        }
    }
    return chars;
}
```

The `</w>` marker is important because it allows the tokenizer to distinguish between a subword that appears at the end of a word versus the same subword in the middle of a word. For example, the sequence `est` at the end of `fastest` should be represented differently from the `est` in `estimating`, because they carry different positional meaning. Without end-of-word markers, the tokenizer would conflate these two cases.

## Stage 2: Training

Training is the heart of BPE. The algorithm iteratively counts the frequency of adjacent symbol pairs in the corpus, finds the most frequent pair, merges it into a new symbol, and repeats until the desired vocabulary size is reached.

The key data structures I use are:

- `vocab`: a mapping from token string to token ID (`std::unordered_map<std::string, int>`) — used for fast lookups during encoding.
- `merges`: an ordered list of merge operations (`std::vector<std::pair<std::string, std::string>>`) — preserves the order in which merges were learned.
- `merge_priority`: maps each merge pair (as a concatenated string) to the iteration index at which it was learned (`std::unordered_map<std::string, int>`), used during encoding to apply merges in the correct order.

The training loop starts by computing pair statistics across all words in the corpus:

```cpp
std::map<std::pair<std::string, std::string>, int> BPETokenizer::get_pair_statistics(
    const std::vector<std::pair<std::vector<std::string>, int>>& word_freqs) {
    std::map<std::pair<std::string, std::string>, int> pair_stats;
    for (const auto& [chars, freq] : word_freqs) {
        for (size_t i = 0; i + 1 < chars.size(); ++i) {
            pair_stats[{chars[i], chars[i + 1]}] += freq;
        }
    }
    return pair_stats;
}
```

I use `std::map` (ordered) for pair statistics rather than `std::unordered_map`, because finding the maximum-frequency pair by value requires iterating over all entries anyway — the ordering is not the bottleneck here, and `std::map` avoids the overhead of designing a hash for `std::pair`.

Once we have the statistics, we find the most frequent pair:

```cpp
std::pair<std::string, std::string> BPETokenizer::get_most_frequent_pair(
    const std::map<std::pair<std::string, std::string>, int>& pair_stats) {
    return std::max_element(pair_stats.begin(), pair_stats.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; })->first;
}
```

We then merge that pair everywhere it appears in the corpus:

```cpp
void BPETokenizer::merge_pair(
    std::vector<std::pair<std::vector<std::string>, int>>& word_freqs,
    const std::pair<std::string, std::string>& best_pair) {
    std::string merged = best_pair.first + best_pair.second;
    for (auto& [chars, freq] : word_freqs) {
        std::vector<std::string> new_chars;
        for (size_t i = 0; i < chars.size(); ++i) {
            if (i + 1 < chars.size() &&
                chars[i] == best_pair.first &&
                chars[i + 1] == best_pair.second) {
                new_chars.push_back(merged);
                ++i;
            } else {
                new_chars.push_back(chars[i]);
            }
        }
        chars = new_chars;
    }
}
```

The `train` method ties it all together:

```cpp
void BPETokenizer::train(const std::string& text, int num_merges) {
    auto pre_tokens = pre_tokenize(text);

    // Build word frequency map
    std::unordered_map<std::string, int> word_freq_map;
    for (const auto& token : pre_tokens) {
        word_freq_map[token]++;
    }

    // Convert words to character sequences with </w> markers
    std::vector<std::pair<std::vector<std::string>, int>> word_freqs;
    for (const auto& [word, freq] : word_freq_map) {
        word_freqs.push_back({word_to_chars(word), freq});
        for (const auto& ch : word_to_chars(word)) {
            if (vocab.find(ch) == vocab.end()) {
                int id = vocab.size();
                vocab[ch] = id;
            }
        }
    }

    // Iteratively learn merges
    for (int i = 0; i < num_merges; ++i) {
        auto pair_stats = get_pair_statistics(word_freqs);
        if (pair_stats.empty()) break;

        auto best_pair = get_most_frequent_pair(pair_stats);
        merge_pair(word_freqs, best_pair);

        std::string merged = best_pair.first + best_pair.second;
        int id = vocab.size();
        vocab[merged] = id;
        merges.push_back(best_pair);
        merge_priority[best_pair.first + " " + best_pair.second] = i;
    }
}
```

To make this concrete, consider training on a corpus containing the word `"hello"` many times. After pre-tokenization and character splitting with end-of-word markers, the word is represented as:

```
Iteration 0: "h e l l o</w>" -> most frequent pair ("l", "l")
Iteration 1: "h e ll o</w>" -> most frequent pair ("e", "ll")
```

After two merges we already have a compact representation of the word. As training continues, increasingly longer subword units are formed — eventually whole common words may become single tokens.

## Stage 3: Encoding

Encoding converts a new piece of text into a sequence of token IDs using the merge rules learned during training. The key method is `apply_bpe`, which applies all learned merges greedily in priority order:

```cpp
std::vector<std::string> BPETokenizer::apply_bpe(
    const std::vector<std::string>& chars) {
    std::vector<std::string> tokens = chars;

    // Keep applying merges until no more can be applied
    bool changed = true;
    while (changed) {
        changed = false;
        int best_priority = INT_MAX;
        int best_idx = -1;

        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            std::string key = tokens[i] + " " + tokens[i + 1];
            auto it = merge_priority.find(key);
            if (it != merge_priority.end() && it->second < best_priority) {
                best_priority = it->second;
                best_idx = i;
            }
        }

        if (best_idx != -1) {
            std::string merged = tokens[best_idx] + tokens[best_idx + 1];
            tokens.erase(tokens.begin() + best_idx + 1);
            tokens[best_idx] = merged;
            changed = true;
        }
    }
    return tokens;
}
```

The greedy strategy applies whichever merge was learned earliest (lowest priority index) at each step, which mirrors how the training corpus was compressed. This ensures that encoding is consistent with what the model saw during training.

The `encode` method pre-tokenizes the input, splits each word into characters, applies BPE, and looks up token IDs:

```cpp
std::vector<int> BPETokenizer::encode(const std::string& text) {
    std::vector<int> ids;
    auto pre_tokens = pre_tokenize(text);
    for (const auto& word : pre_tokens) {
        auto chars = word_to_chars(word);
        auto bpe_tokens = apply_bpe(chars);
        for (const auto& token : bpe_tokens) {
            auto it = vocab.find(token);
            if (it != vocab.end()) {
                ids.push_back(it->second);
            }
        }
    }
    return ids;
}
```

For example, encoding the word `"hello"` after training on a corpus might look like:

```
Input: "hello"
Chars: ["h", "e", "l", "l", "o</w>"]
After merges: ["hell", "o</w>"]
Token IDs: [42, 17]
```

## Stage 4: Decoding

Decoding converts a sequence of token IDs back to the original text. I maintain an `id_to_token` reverse mapping alongside the `vocab` map. Decoding is simply a lookup and string concatenation, with `</w>` markers stripped and replaced by spaces:

```cpp
std::string BPETokenizer::decode(const std::vector<int>& ids) {
    std::string text;
    for (int id : ids) {
        std::string token = id_to_token[id];
        // Replace end-of-word marker with a space
        size_t pos = token.find(END_OF_WORD);
        if (pos != std::string::npos) {
            token.replace(pos, END_OF_WORD.length(), " ");
        }
        text += token;
    }
    // Trim trailing space
    if (!text.empty() && text.back() == ' ') {
        text.pop_back();
    }
    return text;
}
```

Decoding is O(k) in the number of tokens, making it very fast.

# Key Design Decisions

## Greedy merge application

During encoding, I apply merges greedily: at each step, the merge with the lowest priority index (i.e., learned earliest during training) is applied first. This mirrors the order in which merges were discovered from the corpus statistics and produces encodings consistent with the training distribution.

## Vocabulary construction

The vocabulary starts with all unique characters (including the `</w>` variants) found in the training corpus. Each merge operation adds exactly one new token — the concatenation of the two merged symbols. This means that after `m` merges on a corpus with `c` unique characters, the vocabulary size is exactly `c + m`. This is a very predictable and controllable property.

## `std::map` vs `std::unordered_map` for pair statistics

I use `std::map<std::pair<std::string, std::string>, int>` for pair statistics instead of `std::unordered_map`. Since we need to find the maximum-frequency pair anyway (which requires a full scan), the log-factor overhead of `std::map` insertions and lookups is acceptable. Using `std::unordered_map` would require either a custom hash for `std::pair<std::string, std::string>` or wrapping the key — not worth the added complexity for this use case.

For the vocabulary itself, I use `std::unordered_map<std::string, int>` because lookups during encoding are O(1) on average and we do not need ordering.

# Complexity Analysis

## Training

Each training iteration requires:
- Computing pair statistics over the entire corpus: O(n) where n is the total number of tokens in the corpus
- Finding the best pair: O(p) where p is the number of unique pairs
- Merging the pair: O(n) in the worst case

Running `m` merges gives **O(n × m)** total training complexity. For large corpora this can be expensive, but in practice the number of merges is bounded (typically 1,000–50,000) and the per-iteration cost decreases as the vocabulary grows and words become shorter sequences of tokens.

## Encoding

For a single input of length k tokens, applying BPE requires at most m passes (one per merge), and each pass scans the current token sequence. In the worst case this is **O(k × m)**. In practice, most merges do not apply to most inputs, so the average-case complexity is much better.

## Decoding

Decoding is a single pass over the token IDs with a hash map lookup per token: **O(k)**.

# Usage Example

Here is a minimal usage example:

```cpp
BPETokenizer tokenizer;

// Train on a corpus with 500 merge operations
tokenizer.train(corpus_text, 500);

// Encode a string to token IDs
std::vector<int> ids = tokenizer.encode("hello world");

// Decode back to text
std::string text = tokenizer.decode(ids);

// Save vocabulary and merges to disk
tokenizer.save("vocab.txt", "merges.txt");
```

The project also ships a Shakespeare training example that trains the tokenizer on the complete works of Shakespeare and prints some statistics:

```bash
make run-shakespeare
```

This is a convenient way to experiment with different vocabulary sizes and see how the tokenizer handles real literary text.

# Comparison to Other Tokenization Methods

BPE sits in an interesting middle ground between character-level and word-level tokenization:

| Method | Vocabulary Size | Handling OOV | Training Required |
|--------|----------------|--------------|-------------------|
| Character | ~100 | Perfect | No |
| Word | 10,000–100,000 | Poor | No |
| **BPE** | **1,000–50,000** | **Good** | **Yes** |
| WordPiece | 1,000–50,000 | Good | Yes |

- **Character-level** tokenizers have tiny vocabularies and handle any out-of-vocabulary (OOV) input perfectly, but sequences become very long, which is expensive for Transformer models due to the quadratic attention cost.
- **Word-level** tokenizers produce short sequences but have huge vocabularies and fail badly on OOV words (common for names, technical terms, or misspellings).
- **BPE** finds a middle ground: it handles OOV inputs by decomposing unknown words into known subword units, and it produces reasonably compact sequences. The vocabulary size is a tuneable hyperparameter (the number of merges).
- **WordPiece** (used in BERT) is conceptually similar to BPE but selects merges based on a likelihood criterion rather than raw frequency.

# Project Structure

The project is organized as a standard C++17 project with a Makefile-based build system. The core tokenizer lives in `llm/tokenizer.h` and `llm/tokenizer.cpp`, with usage examples under `examples/`. Compiling and running requires only a C++17-capable compiler — no external dependencies.

The Makefile exposes targets for compiling with different optimization levels and running the provided examples, making it straightforward to benchmark the tokenizer on your own corpora.

# Conclusion

In this post I walked through a C++ implementation of the Byte Pair Encoding tokenizer, covering all four stages: pre-tokenization, training, encoding, and decoding. BPE is a beautifully simple algorithm — the core idea of iteratively merging the most frequent pair of symbols is easy to understand yet surprisingly powerful in practice.

Implementing it in C++ from scratch was a great exercise for understanding what happens under the hood in tokenizers like GPT-2's. The key takeaways are:

- The `</w>` end-of-word marker is essential for positional disambiguation of subwords.
- Vocabulary size is directly controlled by the number of merge operations — a clean and interpretable hyperparameter.
- Greedy merge application during encoding is efficient and consistent with the training procedure.
- BPE strikes a practical balance between vocabulary size and sequence length that makes it ideal for large language models.

This tokenizer is part of my ongoing personal C++ project for building LLM components from scratch. I plan to continue extending it with more components — stay tuned for future posts!
