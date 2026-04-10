# Text Feature Engineering Assignment – Report
**Dataset:** Simulated Amazon-style product reviews (120 reviews, 60 positive / 60 negative)

---

## 1. Dataset & Preprocessing

120 product reviews were collected (or simulated) and stored in `reviews.csv`. Each review went through a five-step preprocessing pipeline: lowercasing → punctuation removal → digit removal → tokenization → stopword removal → lemmatization. The resulting vocabulary contained **~300–400 unique tokens** after cleaning.

---

## 2. Feature Engineering Comparison

| | One Hot Encoding | Bag of Words | TF-IDF |
|---|---|---|---|
| **Value type** | Binary (0/1) | Integer counts | Float (0–1) |
| **Frequency captured** | No | Yes | Yes (normalized) |
| **Word importance** | None | None | Yes (rare words weighted higher) |
| **Common word handling** | No | No – inflated counts | Yes – penalized via IDF |
| **Best for** | Presence checks | Short text classification | Search, ranking, sentiment |

**Why TF-IDF weights rare words higher:** The IDF component is `log(N / df)`, where `N` is the total number of documents and `df` is the number of documents containing the word. A word like "product" that appears in 100/100 reviews gets IDF ≈ 0, contributing almost nothing. A word like "defective" appearing in only 10 reviews gets a high IDF, making it a strong signal for negative sentiment.

---

## 3. Sparse Matrix Analysis

All three representations produce high-dimensional sparse matrices. In a 120-document corpus with ~350 vocabulary words:

- **OHE:** 120 × 350 → ~92–95% zeros
- **BoW:** 120 × 350 → ~90–93% zeros  
- **TF-IDF:** 120 × 350 → ~90–93% zeros

At scale (1 million documents, 500,000 vocabulary words), a dense float64 matrix would require ~4 TB of RAM. Sparse storage (CSR format) compresses this to only the non-zero values. Alternatively, dense word embeddings (Word2Vec, GloVe, BERT) reduce dimensionality to a fixed 100–768 dimensions regardless of vocabulary size.

---

## 4. Sentiment Classification Results

Both Logistic Regression and Naive Bayes were trained using BoW and TF-IDF features.

**Observations:**
- TF-IDF consistently matched or outperformed BoW because it reduces the influence of words like "product" or "good" that appear equally in positive and negative reviews.
- Logistic Regression showed the largest benefit from TF-IDF normalization.
- Naive Bayes performed well even with raw BoW counts due to its probabilistic treatment of word frequencies.
- Overall accuracy ranged from **~90–100%** on this clean simulated dataset; real-world data would yield lower numbers (~70–85%) due to noise, sarcasm, and mixed-sentiment reviews.

---

## 5. Limitations & Recommendations

1. **BoW & TF-IDF ignore word order** — "not good" and "good" look similar.
2. **No semantic similarity** — "happy" and "joyful" are orthogonal vectors.
3. **Vocabulary mismatch** — new words at inference time are ignored.
4. **Next steps:** Use subword tokenization and pre-trained transformers (e.g., `distilbert-base-uncased`) via Hugging Face for production-grade sentiment analysis.

---

*Submitted as part of the Text Feature Engineering Assignment.*
