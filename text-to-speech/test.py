import sounddevice as sd
import requests

audio = requests.post("http://127.0.0.1:10000/tts", params={"text": """
Explanation
Convert to NumPy Array: Ensure the audio data is converted to a NumPy array with the correct data type (np.float32).
Check Sampling Rate: Print the shape of the audio array and the sampling rate to verify they are correct.
Play Audio: Use sd.play(audio_array, samplerate=sampling_rate) to play the audio array directly.
Additional Checks
"""})

if audio.status_code != 200:
    print(audio.json())
    exit()

audio = audio.json()

sd.play(audio["audio"], audio["sample_rate"])
sd.wait()