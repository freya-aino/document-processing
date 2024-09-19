import io
import requests

from .chat_gpt import *
from .API import *

# Function to call URLs in the background
def call_url(url, image):
    # Convert image to bytes
    img_byte_arr = io.BytesIO()
    
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Send POST request with image
    with open("huggingface-api-key.txt", "r") as f:
        api_key = f.read().strip()
    return requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        files={'file': img_byte_arr})


def process_image_with_all(image, message_column):
    
    # classification
    classification_response = None
    object_detection_response = None
    pose_estimation_response = None
    text_extraction_response = None
    try:
        # image classificatoin
        response = call_url(IMAGE_CLASSIFICATION, image)
        classification_response = json.loads(response.text)
        message_column.success(f"{response.status_code}: classification")
        
        # object detection
        response = call_url(OBJECT_DETECTION, image)
        object_detection_response = json.loads(response.text)
        if len(object_detection_response["result"]) == 0:
            object_detection_response = None
        message_column.success(f"{response.status_code}: object detection")
        
        # pose estimation
        response = call_url(POSE_ESTIMATION, image)
        pose_estimation_response = json.loads(response.text)
        if len(pose_estimation_response["result"]) == 0:
            pose_estimation_response = None
        message_column.success(f"{response.status_code}: pose estimation")
        
        # text extraction
        response = call_url(OCR_TEXT_EXTRACTION, image)
        text_extraction_response = json.loads(response.text)
        message_column.success(f"{response.status_code}: text extraction")
        
        # text data extraction
        response = call_url(OCR_DATA_EXTRACTION, image)
        text_data_extraction_response = json.loads(response.text)
        message_column.success(f"{response.status_code}: text data extraction")
        
    except Exception as e:
        message_column.warning(f"Error: {e}")
    
    return {
        "classification": classification_response,
        "object_detection": object_detection_response,
        "pose_estimation": pose_estimation_response,
        "text_extraction": text_extraction_response,
        "text_data_extraction": text_data_extraction_response
    }

def get_prompt_formatting(responses):
    return {
        "classification": classification_to_prompt(responses["classification"]) if responses["classification"] else None,
        "object_detection": objects_to_prompt(responses["object_detection"]) if responses["object_detection"] else None,
        "pose_estimation": pose_to_prompt(responses["pose_estimation"]) if responses["pose_estimation"] else None,
        "text_extraction": ocr_to_prompt(responses["text_extraction"], responses["text_data_extraction"]) if responses["text_extraction"] and responses["text_data_extraction"] else None
    }

def gpt_extract_information(responses, message_column):
    
    classification_results = None
    object_detection_results = None
    pose_estimation_results = None
    text_extraction_results = None
    
    chain = EXTRACTION_PROMPT | LLM
    
    formatted_prompts = get_prompt_formatting(responses)
    
    if formatted_prompts["classification"]:
        try:
            classification_results = chain.invoke({
                "instructions": CLASSIFICATION_INSTRUCTIONS, 
                "information": formatted_prompts["classification"]
            }).content
            message_column.success("LLM info for classification")
        except Exception as e:
            message_column.warning(f"Error: {e}")
        
    if formatted_prompts["object_detection"]:
        try:
            object_detection_results = chain.invoke({
                "instructions": OBJECT_DETECTION_INSTRUCTIONS, 
                "information": formatted_prompts["object_detection"]
            }).content
            message_column.success("LLM info for object detection")
        except Exception as e:
            message_column.warning(f"Error: {e}")
        
    if formatted_prompts["pose_estimation"]:
        try:
            pose_estimation_results = chain.invoke({
                "instructions": POSE_INSTRUCTIONS, 
                "information": formatted_prompts["pose_estimation"]
            }).content
            message_column.success("LLM info for pose estimation")
        except Exception as e:
            message_column.warning(f"Error: {e}")
        
    if formatted_prompts["text_extraction"]:
        try:
            text_extraction_results = chain.invoke({
                "instructions": OCR_INSTRUCTIONS, 
                "information": formatted_prompts["text_extraction"]
            }).content
            message_column.success("LLM info for text extraction")
        except Exception as e:  
            message_column.warning(f"Error: {e}")
    
    return {
        "classification": classification_results,
        "object_detection": object_detection_results,
        "pose_estimation": pose_estimation_results,
        "text_extraction": text_extraction_results
    }

def assistant_chat_api(user_input, extracted_information, gpt_extracted_information):
    
    information = ""
    for file_name in gpt_extracted_information:
        gpt_information = gpt_extracted_information[file_name]
        ext_information = extracted_information[file_name]
        
        for k in gpt_information:
            gpt_information[k] = gpt_information[k].replace("\"", "'") if gpt_information[k] else ""
        
        ext_information_prompts = get_prompt_formatting(ext_information)
        
        information += f"""
            <image={file_name}>
                <assistant_information>
                    <classification>{gpt_information.get('classification', '')}</classification>
                    <object_detection>{gpt_information.get('object_detection', '')}</object_detection>
                    <pose_estimation>{gpt_information.get('pose_estimation', '')}</pose_estimation>
                    <text_extraction>{gpt_information.get('text_extraction', '')}</text_extraction>
                </assistant_information>
                <raw_assistant_information>
                    <classification>{ext_information_prompts.get('classification', '')}</classification>
                    <object_detection>{ext_information_prompts.get('object_detection', '')}</object_detection>
                    <pose_estimation>{ext_information_prompts.get('pose_estimation', '')}</pose_estimation>
                    <text_extraction>{ext_information_prompts.get('text_extraction', '')}</text_extraction>
                </raw_assistant_information>
            </image>
            """
    
    chain = CHAT_PROMPT | LLM_CHAT
    
    answer = chain.invoke({
        "instructions": CHAT_INSTRUCTIONS,
        "information": information,
        "prompt": user_input
    }).content
    
    return answer
