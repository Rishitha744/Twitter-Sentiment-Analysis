# Twitter Sentiment Analysis: From TF-IDF to Transformers

This project investigates how well different text representation methods — from 
classical TF-IDF features to modern transformer-based embeddings — can classify 
tweet sentiment, and where each approach succeeds or fails. Using the Sentiment140 
dataset of 1.6 million tweets, we analyze how tweet complexity affects model 
performance and show that DistilBERT's advantage over TF-IDF grows as tweets 
get more complex.

## 👉 Start here: [main_notebook.ipynb](main_notebook.ipynb)

## 🌐 Live Demo
[Try the Streamlit App here](https://twitter-sentiment-analysis-tfjzjs2ybscvcfay2tvdpx.streamlit.app/)

---

## Research Questions

- **RQ1:** How well can traditional TF-IDF based models classify tweet sentiment, 
  and how do different text representations (unigrams vs. bigrams) affect performance?
- **RQ2:** How does text complexity — specifically tweet length and vocabulary 
  diversity — affect sentiment classification performance?
- **RQ3:** Do transformer-based embeddings (DistilBERT) improve sentiment 
  classification compared to TF-IDF baselines, and where do they still fail?

---

## Results Summary

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.7667 | 0.7591 | 0.7765 | 0.7677 |
| DistilBERT (fine-tuned) | 0.8299 | 0.8291 | 0.8280 | 0.8285 |

**Key finding:** DistilBERT outperforms TF-IDF by ~6 points across all metrics. 
More importantly, DistilBERT's advantage *grows* with tweet complexity — gaining 
+0.0479 on short tweets and +0.0764 on long tweets, nearly double the improvement. 
This shows contextual embeddings specifically help where bag-of-words methods 
struggle most.

---

## Dataset

- **Name:** Sentiment140
- **Size:** 1.6 million tweets
- **Labels:** Automatically labeled using emoticons — positive emoticons (`:)`) 
  mark positive tweets and negative emoticons (`:(`)\  mark negative ones. 
  This approach is called *distant supervision*.
- **Source:** https://www.kaggle.com/datasets/kazanova/sentiment140
- **Paper:** Go, Bhayani & Huang (2009). *Twitter Sentiment Classification 
  using Distant Supervision.* Stanford CS224N.

**Preprocessing steps:**
1. Lowercase all text
2. Remove URLs, @mentions, hashtag symbols
3. Collapse repeated characters (e.g. "sooooo" → "soo")
4. Remove duplicate tweets
5. Apply BERTweet-style normalization (URLs → `HTTPURL`, mentions → `@USER`)
6. Stratified sample of 200K tweets (100K positive, 100K negative)
7. 50K stratified subset for DistilBERT fine-tuning

---

## How to Reproduce

This project was built in **Google Colab Pro** with an NVIDIA A100 GPU.

1. Open `main_notebook.ipynb` in Google Colab
2. Upload your `kaggle.json` API key when prompted  
   *(Get it from kaggle.com → Account → Create API Token)*
3. Mount Google Drive when prompted — cleaned data and models are saved there
4. Run all cells in order — the notebook handles all downloads and setup automatically
5. **Important:** Parts 1–4 run on CPU. Part 5 (DistilBERT fine-tuning) requires 
   GPU runtime → *Runtime → Change runtime type → A100 GPU*
6. If your session restarts, use the **Fast Resume cells** in Part 5 to reload 
   models from Drive without retraining

---

## Key Dependencies

| Package | Version |
|---|---|
| Python | 3.11 |
| torch | 2.x |
| transformers | 4.x |
| scikit-learn | 1.x |
| pandas | 2.x |
| numpy | 1.x |
| streamlit | latest |
| gdown | latest |
| wordcloud | latest |
| seaborn | latest |

Full dependency list: [requirements.txt](requirements.txt)

## Repo Structure
Twitter-Sentiment-Analysis/
│
├── main_notebook.ipynb        ← 👉 Final curated notebook (start here)
├── app.py                     ← Streamlit live demo app
├── requirements.txt           ← Full package list exported from Colab
├── .gitignore
├── README.md
│
└── checkpoints/
├── checkpoint_1.ipynb     ← EDA and initial dataset exploration
└── checkpoint_2.ipynb     ← Preprocessing, sampling, and methodology

---

## References

- Go, A., Bhayani, R., & Huang, L. (2009). *Twitter Sentiment Classification 
  using Distant Supervision.* Stanford CS224N Project Report.
- Sanh, V., et al. (2019). *DistilBERT, a distilled version of BERT: smaller, 
  faster, cheaper and lighter.* arXiv:1910.01108.
- Barbieri, F., et al. (2020). *TweetEval: Unified Benchmark and Comparative 
  Evaluation for Tweet Classification.* Findings of EMNLP 2020.
- Nguyen, D., et al. (2020). *BERTweet: A pre-trained language model for 
  English Tweets.* EMNLP 2020.

---

*CSCE 676 Data Mining · Texas A&M University · Sai Rishitha Chittaluru*

---

## Repo Structure
