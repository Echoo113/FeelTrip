import streamlit as st
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 📋 问卷收集函数
def collect_personality():
    st.markdown("## 🧠 Personality Questionnaire")
    st.markdown("Please rate how much you agree with the following statements (1 = Strongly Disagree, 5 = Strongly Agree):")

    questions = [
        "I enjoy meeting new people.",
        "I enjoy spending time alone.",
        "I stay calm under pressure.",
        "I often feel anxious or tense.",
        "I like trying new things.",
        "I’m interested in art or philosophy.",
        "I am a responsible person.",
        "I plan things ahead and stick to deadlines.",
        "I care about others' feelings and enjoy helping.",
        "I prefer cooperation over competition.",
        "I enjoy participating in group activities.",
        "I am good at expressing my ideas.",
        "I like to plan trips in advance.",
        "I’m comfortable making spontaneous travel decisions.",
        "I enjoy exploring unfamiliar places."
    ]

    responses = []
    for idx, q in enumerate(questions, 1):
        val = st.slider(f"Q{idx}: {q}", min_value=1, max_value=5, value=3)
        responses.append(val)

    return responses


# 🧠 个性降维+聚类模型
class PersonalityModel:
    def __init__(self, n_components=3, n_clusters=4):
        self.pca = PCA(n_components=n_components)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.fitted = False

    def fit(self, all_user_vectors):
        # 输入维度：(n_samples, 15)
        self.pca.fit(all_user_vectors)
        pca_output = self.pca.transform(all_user_vectors)
        self.kmeans.fit(pca_output)
        self.fitted = True

    def encode(self, user_vector):
        vec = np.array([user_vector])
        pca_vec = self.pca.transform(vec)
        cluster = self.kmeans.predict(pca_vec)
        return pca_vec.flatten().tolist(), int(cluster[0])
