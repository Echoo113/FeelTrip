import streamlit as st
# -*- coding: utf-8 -*-
st.set_page_config(page_title="FeelTrip", page_icon="🌍")


from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import pandas as pd
import numpy as np
import os
from personality import collect_personality, PersonalityModel
import emoai


if "chat_enabled" not in st.session_state:
    st.session_state.chat_enabled = False

if "emo_history" not in st.session_state:
    st.session_state.emo_history = [] 





def save_user_response(user_vector):
    try:
        file_path = "user_responses.csv"
        columns = [f"q{i+1}" for i in range(15)]
        new_entry = pd.DataFrame([user_vector], columns=columns)

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # delete the first row if it contains the header
            df = df[df["q1"] != "q1"]
            df = pd.concat([df, new_entry], ignore_index=True)
        else:
            df = new_entry

        df.to_csv(file_path, index=False)
        st.success(f"✅ Response saved. File now has {df.shape[0]} records.")
        return True
    except Exception as e:
        st.error(f"❌ Failed to save response: {e}")
        return False

    

   
# model loading
@st.cache_resource
def load_model():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "emotion_model/distilbert-emotion"))
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

# mood detection
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
NEGATIVE_EMOTIONS = {
    "sadness", "anger", "fear", "disappointment", "grief",
    "remorse", "disapproval", "nervousness", "embarrassment"
}

# --------------------------
# Sidebar panel
page = st.sidebar.radio("🧭 Select Mode", ["Emotion Analysis", "Personality Test"])

# --------------------------
# Emotion Analysis 
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
            st.session_state.chat_enabled = True
            st.session_state.emo_history = []
        

            
if st.session_state.chat_enabled:
            st.markdown("---")
            emoai.main()

# --------------------------
if page == "Personality Test":
    st.markdown("<h1 style='text-align: center;'>🧠 Personality Questionnaire</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Rate how well each statement describes you (1 = Strongly Disagree, 5 = Strongly Agree)</p>", unsafe_allow_html=True)

    user_vector = collect_personality()

    # save user response
    def save_user_response(user_vector):
        try:
            file_path = "user_responses.csv"
            columns = [f"q{i+1}" for i in range(15)]
            new_entry = pd.DataFrame([user_vector], columns=columns)

            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df = pd.concat([df, new_entry], ignore_index=True)
            else:
                df = new_entry

            df.to_csv(file_path, index=False)
            return True
        except Exception as e:
            st.error(f"❌ Failed to save response: {e}")
            return False

    # 加载问卷数据
    def load_all_responses():
        try:
            df = pd.read_csv("user_responses.csv")
            df = df[df["q1"] != "q1"]  # 去掉重复表头
            df = df.dropna()
            return df.astype(int).values
        except Exception as e:
            st.warning(f"⚠️ Could not load user data, using fallback: {e}")
            return np.random.randint(1, 6, (50, 15))

    #button to submit personality info
    if st.button("📊 Submit Personality Info"):
        with st.spinner("Saving your response and updating model..."):
            success = save_user_response(user_vector)

            if success:
                
                

                try:
                    df_check = pd.read_csv("user_responses.csv")
                    st.info(f"📝 user_responses.csv 当前共 {df_check.shape[0]} 条记录")
                    real_data = df_check[df_check["q1"] != "q1"].dropna().astype(int).values
                except Exception as e:
                    st.warning(f"⚠️ Could not load user_responses.csv: {e}")
                    real_data = np.random.randint(1, 6, (50, 15))  # fallback for safety

                st.write("📊 Total responses loaded:", real_data.shape[0])  # ✅ 这时候 real_data 已经有了




                # load all responses
                n_clusters = min(4, real_data.shape[0])
                if real_data.shape[0] < 3:
                    st.warning("Not enough responses for PCA training. Need ≥ 3.")
                    st.stop()

                personality_model = PersonalityModel(n_components=3, n_clusters=n_clusters)
                personality_model.fit(real_data)

                # culculate PCA
                pca_vec, cluster = personality_model.encode(user_vector)

                st.markdown("### Encoded Personality Vector:")
                st.write(pca_vec)
                st.markdown(f"### Cluster Assignment: 🎯 Cluster #{cluster}")
            else:
                st.error("Saving failed. Please try again.")


