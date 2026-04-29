import os
from typing import List, Union
from PIL import Image
import requests
from dotenv import load_dotenv

load_dotenv()

class VLMHelper:
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct"):
        """
        Initializes the VLM Helper using Hugging Face Inference API directly via requests.
        """
        self.token = os.getenv("HF_TOKEN")
        if not self.token:
            # We don't raise error here to allow initialization in tests where it might be mocked
            pass
        self.model_name = model_name

    def describe_image(self, image: Image.Image, prompt: str = "Describe this image in detail.") -> str:
        """
        Sends an image to the VLM and returns a text description.
        """
        if not self.token:
             return "HF_TOKEN not set. Cannot call VLM API."
             
        return self._call_hf_vision_api(image, prompt)

    def _call_hf_vision_api(self, image: Image.Image, prompt: str) -> str:
        """
        Helper to call HF Inference API with an image.
        """
        API_URL = f"https://api-inference.huggingface.co/models/{self.model_name}"
        headers = {"Authorization": f"Bearer {self.token}"}

        import io
        import base64
        
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Payload structure for many multi-modal models on HF Inference API
        payload = {
            "inputs": {
                "image": img_str,
                "text": prompt
            }
        }

        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code != 200:
                return f"Error from VLM API: {response.status_code} - {response.text}"
                
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                # Some models return a list of results
                if isinstance(result[0], dict):
                    return result[0].get("generated_text", str(result[0]))
                return str(result[0])
            elif isinstance(result, dict):
                return result.get("generated_text", str(result))
            return str(result)
        except Exception as e:
            return f"VLM API Exception: {str(e)}"

if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1:
        helper = VLMHelper()
        img = Image.open(sys.argv[1])
        description = helper.describe_image(img)
        print(f"Description: {description}")
    else:
        print("Usage: python src/utils/vlm_helper.py <path_to_image>")
