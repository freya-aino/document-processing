import io
import requests

# Function to call URLs in the background
def call_url(url, image):
    # Convert image to bytes
    img_byte_arr = io.BytesIO()
    
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Send POST request with image
    return requests.post(url, files={'file': img_byte_arr})