import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import os  

st.set_page_config(page_title="FeelTrip", page_icon="🌍")





@st.cache_resource
def load_model():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../emotion_model/distilbert-emotion"))
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model




tokenizer, model = load_model()

# Emotion labels
# Reduce the number of labels to 6
EMOTION_LABELS = [
    "sadness", "joy", "love", "anger", "fear", "surprise"
    
]

# Set of emotions considered "negative"
NEGATIVE_EMOTIONS = {
    "sadness", "anger", "fear", "disappointment", "grief",
    "remorse", "disapproval", "nervousness", "embarrassment"
}

# --- UI Setup ---
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

        # Top 1 emotion = optimal label
        primary_idx = top_idx[0]
        primary_emotion = EMOTION_LABELS[primary_idx]
        primary_conf = probs[primary_idx].item()
        is_negative = primary_emotion in NEGATIVE_EMOTIONS

        # 🎯 Optimal label highlight
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

        # Comfort section
        if is_negative:
            st.markdown("---")
            st.markdown("### 💖 You’re Not Alone")
            st.image("https://i.imgur.com/FcGzE18.png", width=200)
            st.info("“Even the darkest night will end and the sun will rise.” – Victor Hugo")
        else:
            st.success("✨ Sounds like you're feeling alright — let’s find a travel spot that fits this vibe!")
