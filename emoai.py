
import streamlit as st
import openai

import os
openai.api_key = os.getenv("OPENAI_API_KEY")

# initialize session state
if "emo_history" not in st.session_state:
    st.session_state.emo_history = []

# initialize chat state
FORBIDDEN = {"math","code","python","c++","java","solve","calculate","algorithm","essay","write","homework","proof","formula"}

def is_emotion_focused(text: str) -> bool:
    return not any(k in text.lower() for k in FORBIDDEN)


def get_ai_response(user_input: str) -> str:
    messages = [
        {"role":"system","content":(
            "You are a supportive AI friend. Provide comforting, empathetic replies only. "
            "If the user asks technical questions, gently redirect back to emotional support."
        )},
        {"role":"user","content":user_input}
    ]
    resp = openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=200
    )
    return resp.choices[0].message.content


def handle_send():
    txt = st.session_state.emo_input.strip()
    if not txt:
        st.warning('Please enter something so we can understand your mood 💬')
    elif not is_emotion_focused(txt):
        st.warning('⚠️ Please focus on your feelings. Avoid technical or mathematical questions.')
    else:
        st.session_state.emo_history.append({'who': 'user', 'text': txt})
        with st.spinner('Thinking...'):
            reply = get_ai_response(txt)
        st.session_state.emo_history.append({'who': 'ai', 'text': reply})

import streamlit as st


def main():
   

    #page definition
    st.markdown("""
    <style>
      
      
      .chat-header {
        flex: 0 0 50px;
        background: #ededed;
        display: flex; align-items: center; justify-content: center;
        font-size: 16px; font-weight: bold;
        border-bottom: 1px solid #ddd;
                
        color: #00F400;
        text-align: center;
        padding: 10px;

      }
      .chat-content {
        flex: 1; padding: 10px;
        overflow-y: auto;
      }
      .message {
        max-width: 75%;
        margin: 6px 0;
        padding: 8px 12px;
        border-radius: 8px;
        line-height: 1.4;
        word-break: break-word;
        clear: both;
        color: #000 
    }
      .message.user {
        background: #dcf8c6; float: right;
        border-bottom-right-radius: 2px;
      }
      .message.ai {
        background: #fff; float: left;
        border-bottom-left-radius: 2px;
        
      }
      
      .input-area input {
        flex: 1;
        border: 1px solid #ccc; border-radius: 18px;
        padding: 8px 12px; font-size: 14px; outline: none;
        transition: border-color 0.3s;
        color: #000;
      }
      .input-area button {
        margin-left: 8px; padding: 8px 16px;
        background: #0084ff; color: #fff;
        border: none; border-radius: 18px;
        font-size: 14px; cursor: pointer;
      }
    </style>
    """, unsafe_allow_html=True)

    
    st.markdown("<div class='chat-wrapper'>", unsafe_allow_html=True)


    st.markdown("<div class='chat-header'>💬 Let's chat with AI</div>", unsafe_allow_html=True)

    #chat history
    st.markdown("<div class='chat-content'>", unsafe_allow_html=True)
    for msg in st.session_state.emo_history:
        cls = 'user' if msg['who']=='user' else 'ai'
        text = msg['text'].replace('\n','<br>')
        st.markdown(f"<div class='message {cls}'>{text}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # bottom input area
    with st.form(key='chat_form', clear_on_submit=True):
        st.markdown("<div class='input-area'>", unsafe_allow_html=True)
        st.text_input('', key='emo_input', placeholder='Type your feelings…')
        st.form_submit_button('send', on_click=handle_send)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
