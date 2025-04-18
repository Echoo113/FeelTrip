# FeelTrip: A Statistical and Algorithmic Travel Recommendation System

## Project Overview

**FeelTrip** is a solo-developed project at the intersection of **statistical modeling**, **machine learning**, and **algorithm design**, enhanced by modern **large language models (LLMs)**. It serves as a functional prototype of an emotion- and personality-aware travel recommendation system, focusing purely on computational approaches without reliance on interdisciplinary psychology. The system utilizes the **OpenAI API** to perform emotion classification using pre-trained transformer models.

---

## Objectives

- **Extract user emotion** via natural language processing (NLP) using LLMs (e.g., DistilBERT via OpenAI API).
- **Map user personality** through questionnaire encoding and unsupervised clustering (e.g., PCA + KMeans).
- **Classify tourism destinations** with multi-label supervised learning using scikit-learn models.
- **Recommend personalized destinations** through vector similarity and content-based filtering.
- **Generate travel routes** using combinatorial optimization techniques (e.g., TSP heuristics).

---

## Highlights

- Emotion classification is powered by OpenAI’s LLMs, accessed via the **OpenAI API**.
- All modeling is done in **Python**, using **PyTorch**, **scikit-learn**, and **pandas**.
- Real-time user interaction is implemented using **Streamlit**, deployed on the cloud.
- Emphasizes **mathematical modeling** and **computational techniques**, making it suitable for CS/stats research portfolios.

## System Architecture

### 1. User Input Encoding

#### Inputs
- Free-form text describing current mood or travel preference.
- A custom Likert-scale questionnaire, yielding a numerical vector.

#### Processing
- **Text**: Tokenized using `spaCy` or `nltk`; transformed via TF-IDF or transformer embeddings.
- **Questionnaire**: Encoded into a vector; optionally reduced with PCA and clustered via KMeans or GMM.

---

## ✅ Current State

### 2. Emotion Recognition Module

This module classifies user input text into emotions using a fine-tuned DistilBERT model.

#### Dataset
- Combined **43,286** samples from:
  - **GoEmotions** (27,286 samples, 27 emotion classes)
  - **Hugging Face Emotion** (16,000 samples, 6 basic emotions)
- Saved in Hugging Face Datasets format (`emotion_model/combined_emotion_dataset/`).

#### Model
- **Architecture**: `distilbert-base-uncased` + classification head
- **Training**: Custom PyTorch training loop (`train_emotion.py`)
  - Epochs: 3, Batch size: 16
  - Optimizer: AdamW (lr = 5e-5)
  - Tokenization: Max length 128 with padding/truncation
- **Output**: Single-label prediction (`argmax`) over 27 emotion classes

#### Files
- `train_emotion.py`: Full training pipeline
- `distilbert-emotion/`: Trained model and tokenizer
- `plot_emotion_distribution.py`: Class distribution plot (`combined_emotion_dist.png`)

---

## ✅ To Do

### 3. Personality Modeling Module

#### Data
- Custom-designed questionnaire with numerical encoding.

#### Methods
- Dimensionality Reduction: PCA to 2–3 dimensions.
- Clustering: KMeans or GMM for user segmentation.

#### Output
- `personality_vector ∈ ℝᵐ`: Numerical embedding of personality traits.

---

### 4. Destination Labeling Module

#### Dataset
- Yelp Open Dataset or TripAdvisor-scraped descriptions and metadata.

#### Labeling Scheme
- Predefined tags (e.g., "relaxing", "nature", "urban", "adventurous").

#### Models
- One-vs-Rest Logistic Regression
- Lasso or ElasticNet classifiers
- Optional: LightGBM with multi-label adaptations

---

### 5. Recommendation Engine

#### Embedding
- User embedding = `concat(emotion_vector, personality_vector)`
- Place embedding = tag vector + TF-IDF or sentence embeddings

#### Similarity Metric
- Cosine similarity for Top-K recommendation

#### Enhancements (Planned)
- Node2Vec / GraphSAGE over user-tag-place graph
- FAISS for fast nearest-neighbor retrieval

---

### 6. Route Planning Module

#### Data
- Manually created or API-generated distance/time matrix

#### Constraints
- Time, location count, budget, or activity preferences

#### Algorithms
- Greedy heuristic
- TSP approximation (2-opt, Simulated Annealing)
- Dynamic Programming (future extension)

#### Output
- Ordered route satisfying user constraints

---

## Frontend Interface

- Built with [Streamlit](https://streamlit.io/)
- Includes:
  - **Input Page**: Mood text + questionnaire
  - **Output Page 1**: Emotion and personality summary
  - **Output Page 2**: Top-K recommended places
  - **Output Page 3**: Travel route visualization

---

## 📊 Datasets Summary

| Module                | Dataset                                      |
|-----------------------|----------------------------------------------|
| Emotion Recognition   | GoEmotions + HF Emotion (merged)            |
| Personality Modeling  | Custom questionnaire                        |
| Destination Labeling  | Yelp Open Dataset / TripAdvisor             |
| Routing               | Custom distance matrix or Google Maps API   |

---

## 🧠 Core Algorithms Summary

| Task                  | Methodology                                   |
|-----------------------|-----------------------------------------------|
| Text Processing       | Tokenization, TF-IDF, Transformer Embeddings  |
| Emotion Classification| DistilBERT (fine-tuned) / TF-IDF Baseline     |
| Personality Clustering| PCA + KMeans / GMM                            |
| Tag Classification    | One-vs-Rest, Lasso, LightGBM                  |
| Recommendation        | Cosine Similarity / Graph Embedding           |
| Route Planning        | Greedy, TSP Approximation, DP (future)        |

---

## 🧭 Project Orientation

- **Focus**: Computational modeling, ML/NLP pipelines, optimization algorithms
- **Avoidance**: Psychological theory or interdisciplinary components
- **Audience**: Researchers and engineers interested in applied AI/ML for travel

---

## 🛠️ Tools & Libraries

- **Language**: Python
- **Core Libraries**: `scikit-learn`, `transformers`, `pandas`, `numpy`, `streamlit`
- **Modeling**: Hugging Face, LightGBM, PCA, clustering
- **Deployment**: Streamlit Cloud (planned)

---

## 📁 Project Structure

```plaintext
FeelTrip/
├── app.py                           
├── requirements.txt              
├── README.md                       

# Core Modules
├── personality/
│   └── questionnaire.py            # Personality questionnaire logic
├── recommender/
│   └── recommend.py                # Emotion/personality-based place matcher
├── planner/
│   └── planner.py                  # Builds and optimizes travel routes
├── utils/
│   └── helpers.py                  # Shared helper functions
├── UI/
│   └── feeltrip_app.py             # Streamlit frontend

# Emotion Recognition Development
├── emotion_model/
│   ├── train_emotion.py                # Fine-tune DistilBERT on emotion data
│   ├── emotion_classifier_baseline.py # TF-IDF + Logistic Regression baseline
│   ├── plot_emotion_distribution.py   # Visualize emotion label distribution
│   ├── merge_goemotions.py            # Merge GoEmotions and HF Emotion datasets
│   ├── combined_emotion_dist.png      # Class distribution plot
│   ├── distilbert-emotion/            # Trained model and tokenizer
│   │   ├── model.safetensors
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   └── ...
│   └── combined_emotion_dataset/      # Hugging Face dataset format
│       ├── data-00000-of-00001.arrow
│       ├── dataset_info.json
│       ├── state.json
│       └── ...
