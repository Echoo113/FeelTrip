# 🌈 FeelTrip: Emotion- and Personality-Aware Travel Recommendation System

> An AI-powered travel recommendation platform that tailors destinations and routes based on users’ emotional states and personality traits.

---

## 🧠 Project Overview

**FeelTrip** is an intelligent and interactive travel recommendation system that analyzes a user’s *current emotions* and *personality* through natural language input and lightweight questionnaires. Based on this analysis, the system suggests travel destinations and personalized routes that match the user’s mood and preferences.

This project combines natural language processing (NLP), statistical modeling, and recommender system techniques. It’s designed as a showcase of interdisciplinary work at the intersection of AI and behavioral understanding.

---

## ✨ Key Features

- 🔍 **Emotion Detection** from user text using machine learning or NLP models  
- 🧭 **Personality Profiling** via short interactive questionnaires  
- 🏞️ **Multi-label Classification** of tourist attractions (e.g., relaxing, adventurous)  
- 🤝 **Customized Recommendations** based on emotion and personality embeddings  
- 🗺️ **Route Planning** with constraints like time and budget  
- 💻 **Interactive Web Interface** built with Streamlit  


---

## 🛠️ Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python  
- **NLP Models**: BERT (via HuggingFace), Naive Bayes, scikit-learn  
- **Statistical Modeling**: PCA, Lasso, Logistic Regression  
- **Recommender Systems**: Content-based filtering, Cosine similarity  
- **Deployment**: Streamlit Cloud  

---

## 📁 Project Structure

```
feeltrip/
├── app.py                  # Main Streamlit application
├── emotion/
│   └── emotion_model.py    # Emotion classification logic
├── personality/
│   └── questionnaire.py    # Personality questionnaire and analysis
├── recommender/
│   └── recommend.py        # Matching and recommendation engine
├── data/
│   └── places.csv          # Labeled dataset of tourist attractions
├── utils/
│   └── helpers.py          # Utility functions
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

---

## 📸 Screenshots (Coming Soon)



---

## 🌐 Live Demo



---

## 🧠 Future Work

---

## 🤝 Acknowledgements

- [GoEmotions Dataset by Google Research](https://github.com/google-research/goemotions)  
- [Yelp Open Dataset](https://www.yelp.com/dataset)  
- [HuggingFace Transformers](https://huggingface.co/transformers/)  

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.
