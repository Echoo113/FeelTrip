import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import pandas as pd
import numpy as np
import os
from personality import collect_personality, PersonalityModel

# Streamlit 页面设置
st.set_page_config(page_title="FeelTrip", page_icon="🌍")

# 模型加载（缓存）
@st.cache_resource
def load_model():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "emotion_model/distilbert-emotion"))
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

# 情绪标签
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
NEGATIVE_EMOTIONS = {
    "sadness", "anger", "fear", "disappointment", "grief",
    "remorse", "disapproval", "nervousness", "embarrassment"
}

# --------------------------
# Sidebar 页面切换
page = st.sidebar.radio("🧭 Select Mode", ["Emotion Analysis", "Personality Test"])

# --------------------------
# Emotion Analysis 页面
if page == "Emotion Analysis":
    st.markdown("<h1 style='text-align: center;'>🌍 FeelTrip: Emotion-Aware Travel Companion</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Tell us how you’re feeling, and we’ll reflect your emotion — and comfort you if needed 💛</p>", unsafe_allow_html=True)

    user_input = st.text_area("📝 How are you feeling today?", height=100, placeholder="Type anything on your mind...")

    if st.button("💡 Analyze My Emotion"):
        if not user_input.strip():
            st.warning("Please enter something so we can understand your mood 💬")
        else:
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).squeeze()
                top_idx = torch.topk(probs, k=5).indices

            primary_idx = top_idx[0]
            primary_emotion = EMOTION_LABELS[primary_idx]
            primary_conf = probs[primary_idx].item()
            is_negative = primary_emotion in NEGATIVE_EMOTIONS

            st.markdown("## 🎯 Optimal Emotion Detected:")
            st.markdown(
                f"<h2 style='text-align: center; color: {'#d9534f' if is_negative else '#5cb85c'};'>"
                f"{primary_emotion.capitalize()} ({primary_conf:.1%})</h2>",
                unsafe_allow_html=True
            )

            if is_negative:
                st.markdown("<p style='text-align: center;'>🔴 Negative Emotion Detected</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='text-align: center;'>🟢 Positive/Neutral Emotion Detected</p>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 🔍 Top 5 Detected Emotions:")
            for idx in top_idx:
                emotion = EMOTION_LABELS[idx]
                confidence = probs[idx].item()
                st.write(f"**{emotion.capitalize()}** — {confidence:.1%}")
                st.progress(confidence)

            if is_negative:
                st.markdown("---")
                st.markdown("### 💖 You’re Not Alone")
                st.image("https://i.imgur.com/FcGzE18.png", width=200)
                st.info("“Even the darkest night will end and the sun will rise.” – Victor Hugo")
            else:
                st.success("✨ Sounds like you're feeling alright — let’s find a travel spot that fits this vibe!")

# --------------------------
# Personality Test 
elif page == "Personality Test":
    

    st.markdown("<h1 style='text-align: center;'>🧠 Personality Questionnaire</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Rate how well each statement describes you (1 = Strongly Disagree, 5 = Strongly Agree)</p>", unsafe_allow_html=True)

    user_vector = collect_personality()  # slider 输入

    # 保存问卷响应的函数
    def save_user_response(user_vector):
        file_path = "user_responses.csv"
        columns = [f"q{i+1}" for i in range(15)]
        new_entry = pd.DataFrame([user_vector], columns=columns)

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df = pd.concat([df, new_entry], ignore_index=True)
        else:
            df = new_entry

        df.to_csv(file_path, index=False)

    # 提交按钮处理逻辑
    if st.button("📊 Submit Personality Info"):
        # Step 1: 保存当前用户响应
        save_user_response(user_vector)

        # Step 2: 读取所有历史数据
        file_path = "user_responses.csv"
        if os.path.exists(file_path):
            real_data = pd.read_csv(file_path).values
        else:
            real_data = np.random.randint(1, 6, (50, 15))  # fallback

        # Step 3: 训练 PCA + 聚类模型
        personality_model = PersonalityModel()
        personality_model.fit(real_data)

        # Step 4: 对当前用户生成向量 + cluster
        pca_vec, cluster = personality_model.encode(user_vector)

        # Step 5: 显示结果
        st.success("✅ Your personality vector has been recorded.")
        st.markdown("### Encoded Personality Vector:")
        st.write(pca_vec)
        st.markdown(f"### Cluster Assignment: 🎯 Cluster #{cluster}")



def save_user_response(user_vector):
    file_path = "user_responses.csv"
    columns = [f"q{i+1}" for i in range(15)]
    new_entry = pd.DataFrame([user_vector], columns=columns)

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = pd.concat([df, new_entry], ignore_index=True)
    else:
        df = new_entry

    df.to_csv(file_path, index=False)