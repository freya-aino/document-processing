import requests

from utils import API

urls = [
    API.IMAGE_CLASSIFICATION,
    API.OBJECT_DETECTION,
    API.POSE_ESTIMATION,
    API.OCR_DATA_EXTRACTION,
    API.OCR_TEXT_EXTRACTION
]


with open("huggingface-api-key.txt", "r") as f:
    api_key = f.read().strip()
headers = {
    "Authorization": f"Bearer {api_key}"
}

test_image = "./test_image.PNG"

for url in urls:
    with open(test_image, "rb") as f:
        files = {"file": f}
        response = requests.post(url, headers=headers, files=files)
        if response.status_code == 200:
            print("success: ", url, "-", response.status_code)
        else:
            print("failed:  ", url, "\n", response.text)
    print("----------------------------------------------------------")


from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os

with open("openai-api-key.txt", "r") as f:
    api_key = f.read().strip()

os.environ["OPENAI_API_KEY"] = api_key

LLM = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2
) 

# Create a prompt template
prompt_template = ChatPromptTemplate("ping")

# Create a chain with the prompt template and the LLM
chain = prompt_template | LLM

# Invoke the model with the prompt
response = chain.invoke({"prompt": "test connection: ping"})

# Print the response
print("chatgpt:", response.content)
print("----------------------------------------------------------")


# # check gradio model

# from gradio_client import Client, handle_file

# with open("./huggingface-api-key.txt", "r") as f:
#     api_token = f.read().strip()
# client = Client("fey-aino/GOT_official_online_demo", hf_token = api_token)
# result = client.predict(
# 		image=handle_file(test_image),
# 		got_mode="plain texts OCR",
# 		fine_grained_mode="box",
# 		ocr_color="red",
# 		ocr_box="Hello!!",
# 		api_name="/run_GOT"
# )
# print(result)
