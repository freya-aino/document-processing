import requests

with open("huggingface-api-key.txt", "r") as f:
    api_key = f.read().strip()

urls = [
    "https://fey-aino-yolov8.hf.space/pose",
    "https://fey-aino-yolov8.hf.space/detection",
    "https://fey-aino-image-classifier.hf.space/predict",
    "https://fey-aino-tesseract.hf.space/extract-text",
    "https://fey-aino-tesseract.hf.space/extract-data",
]
# url = "http://127.0.0.1:10000/predict"

headers = {
    "Authorization": f"Bearer {api_key}"
}

test_image = "C:\\Users\\noone\\Desktop\\receipts\\aerssignal-2024-04-28-202706_003.jpeg"

for url in urls:
    with open(test_image, "rb") as f:
        files = {"file": f}
        response = requests.post(url, headers=headers, files=files)
        if response.status_code == 200:
            print("success: ", url, "\n", response.json())
        else:
            print("failed:  ", url, "\n", response.text)
            print("response headers: ", response.headers)
    print("----------------------------------------------------------")