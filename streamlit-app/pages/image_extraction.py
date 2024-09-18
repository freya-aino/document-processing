import json
import requests
import time
import streamlit as st
from PIL import Image
from PIL import Image, ImageDraw, ImageFont

from utils.url_caller import call_url
from utils.API import IMAGE_CLASSIFICATION, OBJECT_DETECTION, POSE_ESTIMATION, OCR_DATA_EXTRACTION

# Streamlit app
st.set_page_config(layout="wide")
st.title("Image Extractor")

# Initialize session state for uploaded files
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Create a layout with three columns
input_column, output_column, message_column = st.columns([2, 2, 1])

with output_column:
    result_placeholder = st.empty()
with message_column:
    message_placeholder = st.empty()

# Image classification button
with input_column:
    if st.button("Process Images"):
        if st.session_state.uploaded_files:
            with st.spinner("Calling URL..."):
                
                annotated_images = []
                classification_outputs = []
                
                for file in st.session_state.uploaded_files:
                    image = Image.open(file)
                    image.thumbnail((640, 640))
                    
                    # classification
                    classification_response = None
                    try:
                        response = call_url(IMAGE_CLASSIFICATION, image)
                        message_column.success(f"{response.status_code}: classification: {file.name}")
                        
                        classification_response = json.loads(response.text)
                        
                    except requests.exceptions.RequestException as e:
                        message_placeholder.warning(f"Error: {e}")
                    
                    # object detection
                    object_detection_response = None
                    try:
                        response = call_url(OBJECT_DETECTION, image)
                        message_column.success(f"{response.status_code}: object detection: {file.name}")
                        
                        object_detection_response = json.loads(response.text)
                        
                        if len(object_detection_response["result"]) == 0:
                            object_detection_response = None
                        
                    except requests.exceptions.RequestException as e:
                        message_placeholder.warning(f"Error: {e}")
                    
                    # pose estimation
                    pose_estimation_response = None
                    try:
                        response = call_url(POSE_ESTIMATION, image)
                        message_column.success(f"{response.status_code}: pose estimation: {file.name}")
                        
                        pose_estimation_response = json.loads(response.text)
                        
                        if len(pose_estimation_response["result"]) == 0:
                            pose_estimation_response = None
                        
                    except requests.exceptions.RequestException as e:
                        message_placeholder.warning(f"Error: {e}")
                    
                    # text extraction
                    text_extraction_response = None
                    try:
                        response = call_url(OCR_DATA_EXTRACTION, image)
                        
                        message_column.success(f"{response.status_code}: text extraction: {file.name}")
                        
                        text_extraction_response = json.loads(response.text)
                        
                        if len(text_extraction_response["words"]) == 0:
                            text_extraction_response = None
                    
                    except requests.exceptions.RequestException as e:
                        message_placeholder.warning(f"Error: {e}")
                    
                    
                    # ----------------------------------------
                    
                    if classification_response:
                        classification_outputs.append(
                            [
                                f"lemmas: {', '.join(l['lemmas'])} - probability: {p:.2f}"
                                for p, l in zip(classification_response["probabilities"], classification_response["labels"])
                            ]
                        )
                    else:
                        classification_outputs.append([])
                    
                    # ----------------------------------------
                    
                    ann_image = image.copy()
                    draw = ImageDraw.Draw(ann_image)
                    
                    # draw text extraction
                    if text_extraction_response:
                        for word in text_extraction_response["words"]:
                            x, y, w, h = word["bbox"]
                            x *= ann_image.size[0]
                            y *= ann_image.size[1]
                            w *= ann_image.size[0]
                            h *= ann_image.size[1]
                            
                            x1 = x - w / 2
                            y1 = y - h / 2
                            x2 = x + w / 2
                            y2 = y + h / 2
                            
                            draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
                            draw.text((x1 - 5, y1 - 15), word["text"], fill="blue")
                            draw.text((x1 - 5, y1 - 30), f"{word['confidence']:.2f}", fill="blue")
                    
                    # draw pose estimation
                    if pose_estimation_response:
                        for person_landmarks in pose_estimation_response["result"][0]["landmarks"]:
                            for kpt in person_landmarks["keypoints"][0]:
                                if sum(kpt) == 0:
                                    continue
                                draw.circle((kpt[0] * ann_image.size[0], kpt[1] * ann_image.size[1]), 5, fill="red")
                    
                    # draw object detection
                    if object_detection_response:
                        for bbox in object_detection_response["result"][0]["bboxes"]:
                            x, y, w, h = bbox["bbox"]
                            x *= ann_image.size[0]
                            y *= ann_image.size[1]
                            w *= ann_image.size[0]
                            h *= ann_image.size[1]
                            
                            x1 = x - w / 2
                            y1 = y - h / 2
                            x2 = x + w / 2
                            y2 = y + h / 2
                            
                            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
                            draw.text((x1 - 5, y1 - 15), bbox["class"], fill="green")
                            draw.text((x1 - 5, y1 - 30), f"{bbox['confidence']:.2f}", fill="green")
                    
                    annotated_images.append(ann_image)
                
                
                # display results
                with result_placeholder.container():
                    for i in range(len(annotated_images)):
                        st.image(annotated_images[i], use_column_width=True)
                        st.text("\n".join(classification_outputs[i]))
                        st.markdown("---")
                    


# Image upload
with input_column:
    uploaded_files = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    # ensure that only new files are added
    new_files = [file for file in uploaded_files if file not in st.session_state.uploaded_files]
    persisted_old_files = [file for file in st.session_state.uploaded_files if file in uploaded_files]
    st.session_state.uploaded_files = persisted_old_files + new_files

# display images
if st.session_state.uploaded_files:
    with input_column:
        columns = st.columns(3)
        for i, file in enumerate(st.session_state.uploaded_files):
            image = Image.open(file)
            columns[i % 3].image(image, caption=f'{file.name}', use_column_width=True)

