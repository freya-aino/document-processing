import streamlit as st
import requests
from PIL import Image
import io

import streamlit as st

from utils.url_caller import gpt_extract_information, process_image_with_all, assistant_chat_api

# Streamlit app
st.set_page_config(layout="wide")
st.title("Image Assistant")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
    
if "extracted_information" not in st.session_state:
    st.session_state.extracted_information = {}

if "gpt_extracted_information" not in st.session_state:
    st.session_state.gpt_extracted_information = {}

# Create a layout with three columns
input_column, chat_column, output_column, message_column = st.columns([2, 2, 2, 1])

with output_column:
    st.header("Output")
    result_placeholder = st.empty()
with message_column:
    st.header("Messages")
    message_placeholder = st.empty()

# image uplad column
with input_column:
    
    if st.button("Process Images"):
        for file in st.session_state.uploaded_files:
            if file.name in st.session_state.extracted_information:
                continue
            
            image = Image.open(file)
            image.thumbnail((640, 640))
            
            result = process_image_with_all(image, message_column)
            st.session_state.extracted_information[file.name] = result
        
        with result_placeholder.container():
            for result in st.session_state.extracted_information.values():
                st.json(result)
                st.markdown("---")
    
    uploaded_files = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        # ensure that only new files are added
        new_files = [file for file in uploaded_files if file not in st.session_state.uploaded_files]
        persisted_old_files = [file for file in st.session_state.uploaded_files if file in uploaded_files]
        st.session_state.uploaded_files = persisted_old_files + new_files
        
    
    st.button("Clear Extracted Information", on_click=lambda: st.session_state.extracted_information.clear())
    
    # display images
    if st.session_state.uploaded_files:
        columns = st.columns(2)
        for i, file in enumerate(st.session_state.uploaded_files):
            image = Image.open(file)
            columns[i % 2].image(image, caption=f'{file.name}', width=150)


# chat functionality
with chat_column:
    
    if st.session_state.extracted_information:
        
        if st.button("GPT Extract Information"):
            for file_name, result in st.session_state.extracted_information.items():
                if file_name in st.session_state.gpt_extracted_information:
                    continue
                
                gpt_extracted_information = gpt_extract_information(result, message_column)
                st.session_state.gpt_extracted_information[file_name] = gpt_extracted_information
            
            with result_placeholder.container():
                for result in st.session_state.gpt_extracted_information.values():
                    st.json(result)
                    st.markdown("---")
        
        if st.session_state.gpt_extracted_information:
            
            user_input = st.chat_input("Start chatting about your images here...")
            
            with st.spinner("Assistant is processing..."):
                initial_introduction = assistant_chat_api("Hi, please introduce yourselfe, state your purpose and mention a quick one sentence reference to the images the data provided describe.", st.session_state.extracted_information, st.session_state.gpt_extracted_information)
                st.session_state.chat_history.append({"role": "assistant", "content": initial_introduction})
            
            if user_input:
                
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                with st.spinner("Assistant is processing..."):
                    response = assistant_chat_api(user_input, st.session_state.extracted_information, st.session_state.gpt_extracted_information)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            for message in reversed(st.session_state.chat_history):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        st.button("Clear GPT Extracted Information", on_click=lambda: st.session_state.gpt_extracted_information.clear())
        
