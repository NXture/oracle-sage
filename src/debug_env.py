import os
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("QDRANT_URL")
print(f"RAW URL: '{url}'") # The single quotes will reveal if there are hidden spaces!