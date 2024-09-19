import json
import requests
import time
import streamlit as st
from PIL import Image
from PIL import Image, ImageDraw, ImageFont

from utils.url_caller import *
from utils.chat_gpt import *

# Streamlit app
st.set_page_config(layout="wide")
st.title("Image generator")

# Initialize session state for uploaded files
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
    
if "extracted_information" not in st.session_state:
    st.session_state.extracted_information = {}
    
if "gpt_extracted_information" not in st.session_state:
    st.session_state.gpt_extracted_information = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_image_prompt" not in st.session_state:
    st.session_state.current_image_prompt = ""

if "result_images_urls" not in st.session_state:
    st.session_state.result_images_urls = []

# Create a layout with three columns
input_column, output_column, message_column = st.columns([2, 2, 1])

with output_column:
    st.header("Output")
with message_column:
    st.header("Messages")

with output_column:
    if st.session_state.current_image_prompt != "":
        st.markdown(st.session_state.current_image_prompt)
        st.markdown("---")
        
        image_size = st.number_input("Image Size", value=1024)
        number_of_images = st.number_input("Number of Images", min_value=1, max_value=10, value=1)
        
        if st.button("Generate Images"):
            image_urls = generate_image(st.session_state.current_image_prompt, size=(image_size, image_size), n=number_of_images)
            st.session_state.result_images_urls = image_urls
    
    if st.session_state.result_images_urls != []:
        for url in st.session_state.result_images_urls:
            st.image(url, use_column_width=True)

# Image upload
with input_column:
    
    if st.session_state.gpt_extracted_information and st.session_state.extracted_information:
        
        user_prompt = st.text_area("Generation Prompt")
        
        if st.button("Generate Prompt"):
            
            image_prompt = image_generator_api(
                user_prompt,
                st.session_state.extracted_information,
                st.session_state.gpt_extracted_information,
                st.session_state.chat_history
            )
            st.session_state.current_image_prompt = image_prompt
            
            st.rerun()
    
    
    if st.button("Extract Information"):
        for file in st.session_state.uploaded_files:
            image = Image.open(file)
            image.thumbnail((640, 640))
            
            with message_column.container():
                if file.name not in st.session_state.extracted_information:
                    extraction_results = process_image_with_all(image, message_column)
                    st.session_state.extracted_information[file.name] = extraction_results
                
                if file.name not in st.session_state.gpt_extracted_information:
                    gpt_results = gpt_extract_information(extraction_results, message_column)
                    st.session_state.gpt_extracted_information[file.name] = gpt_results
            
        # Force a rerun to update the UI
        st.rerun()
    
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
            columns[i % 3].image(image, caption=f'{file.name}', use_column_width=True)
            
    if st.button("Clear Chat", type="secondary"):
        st.session_state.chat_history.clear()
    if st.button("Clear Extracted Information", type="secondary"):
        st.session_state.extracted_information.clear()
    if st.button("Clear GPT Extracted Information", type="secondary"):
        st.session_state.gpt_extracted_information.clear()
    