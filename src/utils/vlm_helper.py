import io
import os
import time
import socket
from typing import List, Union

import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv()


class VLMHelper:
    def __init__(self, model_name: str = None):
        """
        Initializes the VLM Helper using Hugging Face Inference API.
        """
        self.token = (os.getenv("HF_TOKEN") or "").strip()
        self.model_name = (model_name or os.getenv(
            "HF_VLM_MODEL", "Salesforce/blip-image-captioning-base"
        )).strip()
        
        # Diagnostic: Print masked token and resolve IP
        if self.token:
            masked = f"{self.token[:5]}...{self.token[-3:]}"
            print(f"DEBUG: HF_TOKEN detected: {masked}")
        else:
            print("DEBUG: HF_TOKEN IS EMPTY!")
            
        try:
            ip = socket.gethostbyname("api-inference.huggingface.co")
            print(f"DEBUG: api-inference.huggingface.co resolved to {ip}")
        except Exception as e:
            print(f"DEBUG: DNS Resolution failed: {e}")

    def describe_image(
        self, image: Image.Image, prompt: str = "Describe this image."
    ) -> str:
        """
        Sends an image to the VLM and returns a text description.
        """
        if not self.token:
            return "HF_TOKEN not set. Cannot call VLM API."

        return self._call_hf_vision_api(image, prompt)

    def _call_hf_vision_api(self, image: Image.Image, prompt: str) -> str:
        """
        Helper to call HF Inference API with raw requests and diagnostics.
        """
        import io

        buffered = io.BytesIO()
        # Scale down for faster processing and lower payload
        if max(image.size) > 600:
            image.thumbnail((600, 600))
        
        image.save(buffered, format="JPEG")
        image_data = buffered.getvalue()

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/octet-stream",
        }
        
        # Primary model and a fallback
        models = [self.model_name, "Salesforce/blip-image-captioning-base"]
        
        for model in models:
            # Try two different URL formats
            urls = [
                f"https://api-inference.huggingface.co/models/{model}",
                f"https://api-inference.huggingface.co/pipeline/image-to-text/{model}"
            ]
            
            for url in urls:
                try:
                    print(f"DEBUG: POSTing to {url}")
                    response = requests.post(
                        url, headers=headers, data=image_data, timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if isinstance(result, list) and len(result) > 0:
                            return result[0].get("generated_text", str(result[0]))
                        return result.get("generated_text", str(result))
                    
                    elif response.status_code == 503:
                        print(f"Model {model} loading (503). Waiting 15s...")
                        time.sleep(15)
                        # Retry once for 503
                        response = requests.post(url, headers=headers, data=image_data, timeout=30)
                        if response.status_code == 200:
                            result = response.json()
                            return result[0].get("generated_text", str(result[0]))
                    
                    print(f"DEBUG: {url} returned {response.status_code}")
                    if response.status_code == 404 and "Cannot POST" in response.text:
                        print("CRITICAL: Detected local proxy/router interference (Express.js error style).")
                        
                except Exception as e:
                    print(f"DEBUG: Request failed: {repr(e)}")
                    
        return "Error: VLM API failed. Please check if a local firewall or proxy is blocking POST requests to huggingface.co"


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        helper = VLMHelper()
        if os.path.exists(sys.argv[1]):
            img = Image.open(sys.argv[1])
            print(f"Result: {helper.describe_image(img)}")
        else:
            print("File not found.")
