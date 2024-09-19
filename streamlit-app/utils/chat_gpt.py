import json
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# ------------------------------------------------------------------------------
with open("./openai-api-key.txt", "r") as f:
    openai_api_key = f.read().strip()

os.environ["OPENAI_API_KEY"] = openai_api_key

# ------------------------------------------------------------------------------

LLM = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

LLM_CHAT = ChatOpenAI(
    model="gpt-4o",
    temperature=0.05,
    max_tokens=None,
    timeout=None,
    max_retries=2
)


EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "<instructions>{instructions}</instructions>"
        ),
        (
            "assistant",
            "<information>{information}</information>"
        )
    ]
)

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "<instructions>{instructions}</instructions>\n<information>{information}</information>"
        ),
        (
            "user",
            "<prompt>{prompt}</prompt>"
        )
    ]
)

# ------------------------------------------------------------------------------

CHAT_INSTRUCTIONS = """
you are charlie, a personal, ready to help, sevice chatbot to help the user with their prompts.
The user uploads a set of images that they wish to talk and inqurry information about.

Reason over each of the images information and provide the user with a personal answer to their questions.
Answer in a polite, professional but kind manner, and explain why you are providing the information you are providing.
Keep the answers short and differentiate between the user asking:
- a techical (extraction) question, for example "how many people are there accross all the images?" or "how many document images are there ?"
- or a general (explanation) question, for example "what motives do we have in the images?" or "what are the images about?"

you need to use the following specifications about the chat to answer the questions and describe the information:
- your assistants have a good understanding of each piece of information, their assesstment should be considered accurate and reliable.
- only in edge cases, where information between assistants is conflicting, you should re-evaluate using the raw_assistant_information to provide answers.
- <information> provides two sets of information for each image in the collection:
    1) <assistant_information> - a set of proffessional assistants extracted usefull descriptions and information.
    2) <raw_assistant_information> - the raw information extracted directly from the images.
- the information provided is in the format:
    [<image=file_name>
        <assistant_information>
            <classification>...</classification>
            <object_detection>...</object_detection>
            <pose_estimation>...</pose_estimation>
            <text_extraction>...</text_extraction>
        </assistant_information>
        <raw_assistant_information>
            <classification>...</classification>
            <object_detection>...</object_detection>
            <pose_estimation>...</pose_estimation>
            <text_extraction>...</text_extraction>
        </raw_assistant_information>
    </image>, <image=file_name>...</image>, ...]
- the designated information is in the format:
    <classification>...</classification> - classification information extracted from the image
    <object_detection>...</object_detection> - object detection information extracted from the image
    <pose_estimation>...</pose_estimation> - pose estimation information extracted from the image
    <text_extraction>...</text_extraction> - text extraction information extracted from the image
- not all information is available for all images, so some information might be missing, for example an image without persons, pose_estimation_information should be empty.
"""

OCR_INSTRUCTIONS = """
you are a optical character recognition (ocr) parsing expert.
in a 1 paragraph answer describe the information received.
reason over the ocr text received and the word bits to answer questions about:
how many seperate texts there are, what the texts are about, where the texts are in the image and what type of texts it is (document, receipt, billboard, etc).

Text extraction might be missleading, so reason over the position and location of each word bit to determine if the text is a document or appears in a non document image.
If you determine it is most likely a document you should list all available information in the structure you think is displayed: if it is a list, a table, a paragraph, etc..

you need to use the following specifications about the ocr to answer the questions and describe the information:
- the ocr text is in the format: ocr_text=text.
- the ocr text is the result of ocr being performed on the image, so spacing and formatting is not reliable.
- when ocr is applied to images without text, the ocr will still return text, but it will be not legible and not useable.
- the ocr word informations are in the format: ocr_word_bits=words.
- each element in the ocr_word_bits has the following data:
    - text: the text of the word bit.
    - bbox: the bounding box of the word bit in the format: [x, y, w, h] where x and y are the top left corner of the bounding box, and w and h are the width and height of the bounding box.
    - confidence: confidence of the word being detected between 0 and 1, where 1 is the highest confidence.
"""

OBJECT_DETECTION_INSTRUCTIONS = """
you are a object detection parsing expert.
in a 1 paragraph answer describe the information received.
reason over the objects names, their locations in the image, their size and the confidence of their prediction to answer quetions about: 
what the objects are doing, where they are in te image and how they might be interacting with each other.

you need to use the following specifications about the objects to answer the questions and describe the information:
- the objects are designated by their class name designated by class=class_name
- the bounding box of the object is in the format: bbox=[x, y, w, h] where x and y are the top left corner of the bounding box, and w and h are the width and height of the bounding box
- the x, y and w, h values are between 0 and 1 and represent the position and size of the bounding box in the image
- image coordinates are in the format: x=0 is the left side of the image, x=1 is the right side of the image, y=0 is the top of the image, y=1 is the bottom of the image
- the confidence values are between 0 and 1 and represent the confidence of the object being detected
"""

POSE_INSTRUCTIONS = """
you are a image pose parsing expert. 
in a 1 paragraph answer describe the information received. 
reason over all keypoints and the confidence of each keypoint and answer questions about: 
how many people there are, each persons pose, where the persons are in the image and what they might be doing.

you need to use the following specifications about the pose to answer the questions and describe the information:
- the keypoints are in the format: name=[x, y] listed in the keypoints list
- each keypoint has a confidence value formatted as name=confidence
- the keypoint values are between 0 and 1 and represent the position of the keypoint in the image
- the directions of the image relative to the keypoints are: x=0 is the left side of the image, x=1 is the right side of the image, y=0 is the top of the image, y=1 is the bottom of the image
- the confidence values are between 0 and 1 and represent the confidence of the keypoint being detected
"""

CLASSIFICATION_INSTRUCTIONS = """
you are a image classification parsing expert. 
in a 1 paragraph answer describe the information received.
reson over all lemmas and probabilities and answer questions about:
what the image is about, what type of image or document it is and what the different probabilities indicate the image content to be.

you need to use the following specifications about the classification to answer the questions and describe the information:
- each row is a classification result with the lemmas and probabilities
- the lemmas are in the format: lemmas=lemma1, lemma2, ..., lemmaN and denote a list of possible words to describe that image classification
- the information starts with the dataset name used for classification
- the probabilities are in the format: probability=0.0 and denote the probability of that classification being correct
- when classification probabilities are spread out over multiple classes, the image is likely to be a mix of those classes.
- similar classes might appear as multiple slightly different individual classes and probabilities, this can make their combined probability higher that other classes.
"""

# ------------------------------------------------------------------------------

def classification_to_prompt(classification_response):
    out = "\n".join([
        f"lemmas= {', '.join(l['lemmas'])} - probability: {p:.2f}"
        for p, l in zip(classification_response["probabilities"], classification_response["labels"])
    ])
    return "imagenet 21k image classification:\n" + out

def pose_to_prompt(pose_response):
    keypoint_names = ["Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"]
    out = ""
    for (i, person_landmarks) in enumerate(pose_response["result"][0]["landmarks"]):
        out += f"person {i}:\n"
        out += "keypoints: " + ", ".join(f"{name}={kpt}" for (name, kpt) in zip(keypoint_names, person_landmarks["keypoints"][0])) + "\n"
        out += "confidences: " + ", ".join(f"{name}={kpt}" for (name, kpt) in zip(keypoint_names, person_landmarks["confidence"][0])) + "\n"
    return "human poses:\n" + out

def objects_to_prompt(objects_response):
    out = ""
    for object_ in objects_response["result"][0]["bboxes"]:
        out += f"class={object_['class']}" + "\n"
        out += f"bbox={object_['bbox']}" + "\n"
        out += f"confidence={object_['confidence']}" + "\n"
    return "objects:\n" + out

def ocr_to_prompt(ocr_response_text, ocr_response_data):
    out = "ocr_text:\n" + ocr_response_text["text"] + "\n"
    out += "ocr_word_bits:\n" + json.dumps(ocr_response_data["words"]) + "\n"
    return "optical_character_recognition (ocr):\n" + out
