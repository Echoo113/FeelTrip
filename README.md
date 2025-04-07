# FeelTrip: A Statistical and Algorithmic Travel Recommendation System

## Project Overview

FeelTrip is a solo-developed project that focuses on the intersection of **statistical modeling**, **machine learning**, and **algorithm design**. It is a functional prototype for an emotion- and personality-aware travel recommendation system, built entirely without interdisciplinary collaboration. The emphasis is on mathematical modeling, data processing, and computational algorithms—psychological framing is intentionally minimized.

---

## Objectives

- Extract user emotion via natural language processing.
- Map user personality through numerical questionnaire encoding and clustering.
- Classify tourism destinations using multi-label supervised learning.
- Match users to destinations using similarity-based recommendation algorithms.
- Construct personalized, constraint-based travel routes via combinatorial optimization.

---

## System Architecture

### 1. User Input Encoding

#### Inputs
- Free-form text describing current mood or travel preference.
- A custom-designed questionnaire (Likert scale), yielding a vector representation.

#### Processing
- Text: tokenized using `spaCy` or `nltk`, transformed with TF-IDF or transformer embeddings.
- Questionnaire: vectorized, optionally reduced via PCA and clustered (KMeans or GMM).

---

### 2. Emotion Recognition Module

#### Dataset
- [GoEmotions (Google Research)](https://github.com/google-research/google-research/tree/master/goemotions) — 58k labeled Reddit comments with 27 emotion classes.

#### Algorithms
- Baseline: TF-IDF + Multilabel Logistic Regression / Naive Bayes.
- Advanced: Fine-tuned `bert-base-uncased` or `distilbert-base-uncased` via HuggingFace Transformers.

#### Output
- `emotion_vector ∈ ℝⁿ`: multilabel probability distribution of emotions.

---

### 3. Personality Modeling Module

#### Data
- Custom questionnaire with numerical encoding.

#### Methods
- Dimensionality Reduction: PCA to 2–3 dimensions.
- Clustering: KMeans or GMM to group user types.

#### Output
- `personality_vector ∈ ℝᵐ`: numerical personality embedding.

---

### 4. Destination Labeling Module

#### Dataset
- Yelp Open Dataset or scraped TripAdvisor data.
- Each destination has textual description, metadata (rating, category).

#### Labeling Scheme
- Predefined tags (e.g., "relaxing", "nature", "urban", "adventurous").

#### Models
- Multi-label Logistic Regression (One-vs-Rest).
- Lasso or ElasticNet regularized classifiers.
- Optional: LightGBM multi-label adaptation.

---

### 5. Recommendation Engine

#### Embedding
- User embedding = `concat(emotion_vector, personality_vector)`.
- Place embedding = tag vector + TF-IDF or sentence embedding.

#### Similarity Metric
- Cosine similarity for top-K recommendation.

#### Optional Enhancements
- Node2Vec/GraphSAGE over a user-tag-place graph for graph-based embeddings.
- FAISS for fast approximate nearest-neighbor search.

---

### 6. Route Planning Module

#### Data
- Manually constructed or API-generated distance/time matrix between destinations.

#### Constraints
- Max total time, max number of locations, or total budget.

#### Algorithms
- Greedy approximation.
- TSP approximation: 2-opt, Simulated Annealing.
- Dynamic Programming (future extension).

#### Output
- Ordered list of destinations satisfying user constraints.

---

## Frontend Interface

- Built using [Streamlit](https://streamlit.io/).
- Modules:
  - Input Page: text + questionnaire.
  - Output Page 1: Emotion and Personality summary.
  - Output Page 2: Top-K recommended places with similarity scores.
  - Output Page 3: Route map with total estimated time.

---

## Datasets Summary

| Module             | Dataset                                              |
|--------------------|------------------------------------------------------|
| Emotion Recognition | GoEmotions                                            |
| Personality Modeling | Custom-designed questionnaire                        |
| Destination Tags    | Yelp Open Dataset / TripAdvisor                       |
| Routing             | Distance matrix (manual or via Google Maps API)      |

---

## Core Algorithms Summary

| Task                | Methodology                                           |
|---------------------|--------------------------------------------------------|
| Text Preprocessing  | Tokenization, TF-IDF, Transformer Embedding           |
| Emotion Classification | Multilabel Logistic Regression / BERT               |
| Personality Profiling | PCA + Clustering (KMeans / GMM)                     |
| Tag Classification  | One-vs-Rest, Lasso, LightGBM                         |
| Recommendation      | Cosine Similarity / Graph Embedding                  |
| Route Planning       | Greedy Search, TSP Approximation, Constraint Handling|

---

## Project Orientation

- **Disciplinary Focus**: Mathematical modeling, algorithm implementation, statistical learning.
- **Exclusion**: No psychological interpretation or interdisciplinary modeling.
- **Target Audience**: Researchers, engineers, and developers interested in practical applications of ML/NLP in recommendation systems.

---

## Implementation Tools

- **Language**: Python
- **Libraries**: scikit-learn, transformers, pandas, numpy, Streamlit
- **Modeling**: HuggingFace, LightGBM, PCA, clustering
- **Deployment**: Streamlit Cloud / Render

---

## Repository Structure

```
feeltrip/
├── app.py
├── emotion/
│   └── emotion_model.py
├── personality/
│   └── questionnaire.py
├── recommender/
│   └── recommend.py
├── planner/
│   └── planner.py
├── data/
│   └── places.csv
├── utils/
│   └── helpers.py
├── requirements.txt
└── README.md
```
