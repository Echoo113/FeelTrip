# personality_form.py

import streamlit as st

def collect_personality():
    st.markdown("##  Personality Questionnaire")
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
