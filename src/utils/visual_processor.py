import os
import cv2
from PIL import Image
from typing import List, Dict, Any

class VisualProcessor:
    def __init__(self):
        """
        Initializes the VisualProcessor for image and video handling.
        """
        pass

    def extract_frames(self, video_path: str, fps: int = 1) -> List[Dict[str, Any]]:
        """
        Extracts frames from a video at a fixed interval.
        Returns a list of dicts with frame data and metadata (timestamp).
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        if video_fps == 0:
            raise ValueError(f"Could not determine FPS for video: {video_path}")

        hop = max(1, round(video_fps / fps))
        frames = []
        count = 0
        success = True

        while True:
            success, frame = cap.read()
            if not success:
                break
            
            if count % hop == 0:
                # Convert BGR (OpenCV) to RGB (PIL/VLM)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                timestamp = count / video_fps
                
                frames.append({
                    "timestamp": timestamp,
                    "image": pil_img,
                    "frame_index": count
                })
            count += 1

        cap.release()
        return frames

    def process_image(self, image_path: str) -> Image.Image:
        """
        Loads and prepares an image for processing.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        return Image.open(image_path).convert("RGB")

if __name__ == "__main__":
    # Quick test if run directly
    import sys
    if len(sys.argv) > 1:
        processor = VisualProcessor()
        path = sys.argv[1]
        try:
            if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                print(f"Extracting frames from video: {path}...")
                frames = processor.extract_frames(path)
                print(f"Extracted {len(frames)} frames.")
                if frames:
                    print(f"First frame timestamp: {frames[0]['timestamp']}s")
            else:
                print(f"Processing image: {path}...")
                img = processor.process_image(path)
                print(f"Image size: {img.size}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python src/utils/visual_processor.py <path_to_file>")
