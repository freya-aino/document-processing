import streamlit as st
import requests
from PIL import Image
import io

import streamlit as st

from utils.url_caller import assistant_chat_api

# Streamlit app
st.set_page_config(layout="wide")
st.title("Image Assistant")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Create a layout with three columns
input_column, chat_column, output_column, message_column = st.columns([2, 2, 2, 1])

with output_column:
    result_placeholder = st.empty()
with message_column:
    message_placeholder = st.empty()


# image uplad column
with input_column:
    st.header("Upload Images")
    
    uploaded_files = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        # ensure that only new files are added
        new_files = [file for file in uploaded_files if file not in st.session_state.uploaded_files]
        persisted_old_files = [file for file in st.session_state.uploaded_files if file in uploaded_files]
        st.session_state.uploaded_files = persisted_old_files + new_files
        
    # display images
    if st.session_state.uploaded_files:
        columns = st.columns(3)
        for i, file in enumerate(st.session_state.uploaded_files):
            image = Image.open(file)
            columns[i % 3].image(image, caption=f'{file.name}', width=50)

# chat functionality

with chat_column:
    st.header("Assistant")
    
    if uploaded_files:
        user_input = st.chat_input("Start chatting about your images here...")
        
        if user_input:
            
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            with st.spinner("Assistant is processing..."):
                response = assistant_chat_api(user_input, st.session_state.uploaded_files)
                st.session_state.chat_history.append({
                    "role": "api",
                    "content": response
                })
            
    for message in reversed(st.session_state.chat_history):
        if message["role"] == "user":
            st.markdown(f"""
                <div style="text-align: right; background-color: rgba(0, 127, 255, 0.1); padding: 10px; border-radius: 10px; margin: 5px;">
                    {message["content"]}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="text-align: left; background-color: rgba(255, 0, 127, 0.1); padding: 10px; border-radius: 10px; margin: 5px;">
                    {message["content"]}
                </div>
            """, unsafe_allow_html=True)
    
