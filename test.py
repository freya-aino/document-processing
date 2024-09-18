import requests

with open("huggingface-api-key.txt", "r") as f:
    api_key = f.read().strip()

url = "https://fey-aino-yolov8.hf.space/pose"
# url = "http://127.0.0.1:10000/predict"
headers = {
    "Authorization": f"Bearer {api_key}"
}

test_image = "C:\\Users\\noone\\Desktop\\receipts\\aerssignal-2024-04-28-202706_003.jpeg"

with open(test_image, "rb") as f:
    files = {"file": f}
    response = requests.post(url, headers=headers, files=files)

print(response)