import sys

from dotenv import load_dotenv

from src.pipeline.query_engine import MultiModalQueryEngine


def run_query(query_text: str):
    """
    Runs a query through the MultiModalQueryEngine and prints the results.
    """
    load_dotenv()

    # Optional: Customize weights via env or logic
    # weights = {"audio": 1.5, "image": 1.0, "video_frame": 1.2}

    try:
        engine = MultiModalQueryEngine()
        response = engine.query(query_text)

        print("\n" + "=" * 50)
        print(f"QUERY: {query_text}")
        print("=" * 50)
        print(f"RESPONSE:\n{response}")
        print("=" * 50 + "\n")

    except Exception as e:
        print(f"Error running query: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query_str = " ".join(sys.argv[1:])
        run_query(query_str)
    else:
        usage = "Usage: python scripts/query_pipeline.py \"Query string\""
        print(usage)
