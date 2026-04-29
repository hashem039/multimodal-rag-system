import os
from src.pipeline.visual_pipeline import VisualPipeline
from src.pipeline.core import init_settings
from dotenv import load_dotenv

load_dotenv()

def main():
    init_settings()
    pipeline = VisualPipeline()

    # Paths to sample data
    image_dir = "data/images"
    video_dir = "data/video"

    # Process images
    if os.path.exists(image_dir):
        for img_file in os.listdir(image_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(image_dir, img_file)
                print(f"\n--- Ingesting Image: {img_file} ---")
                try:
                    nodes = pipeline.process_image(img_path)
                    print(f"Indexed image: {img_file}")
                    print(f"Description: {nodes[0].text[:100]}...")
                except Exception as e:
                    print(f"Error processing image {img_file}: {e}")

    # Process videos
    if os.path.exists(video_dir):
        for vid_file in os.listdir(video_dir):
            if vid_file.lower().endswith(('.mp4', '.avi', '.mov')):
                vid_path = os.path.join(video_dir, vid_file)
                print(f"\n--- Ingesting Video: {vid_file} ---")
                try:
                    # Sampling at 1fps as per requirements
                    nodes = pipeline.process_video(vid_path, fps=1)
                    print(f"Indexed {len(nodes)} frames from video: {vid_file}")
                except Exception as e:
                    print(f"Error processing video {vid_file}: {e}")

if __name__ == "__main__":
    main()
