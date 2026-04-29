import os
from typing import List, Union
from PIL import Image
import requests
from dotenv import load_dotenv

load_dotenv()

class VLMHelper:
    def __init__(self, model_name: str = "nlpconnect/vit-gpt2-image-captioning"):
        """
        Initializes the VLM Helper using Hugging Face Inference API directly via requests.
        """
        self.token = os.getenv("HF_TOKEN")
        self.model_name = model_name

    def describe_image(self, image: Image.Image, prompt: str = "Describe this image.") -> str:
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
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_data = buffered.getvalue()

        try:
            # Try simple data payload first
            response = requests.post(API_URL, headers=headers, data=image_data, timeout=30)
            
            if response.status_code != 200:
                # Try JSON payload as fallback
                import base64
                img_str = base64.b64encode(image_data).decode()
                payload = {"inputs": img_str}
                response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
                
            if response.status_code != 200:
                return f"Error from VLM API: {response.status_code} - {response.text[:100]}"
                
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict):
                    return result[0].get("generated_text", result[0].get("caption", str(result[0])))
                return str(result[0])
            elif isinstance(result, dict):
                return result.get("generated_text", result.get("caption", str(result)))
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
