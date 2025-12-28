from qdrant_client import QdrantClient
from langchain_google_genai import ChatGoogleGenerativeAI

print("--- 🩺 HARDCODED CONNECTION TEST ---")

# 1. I cleaned your URL (Removed :6333 and spaces)
CLEAN_URL = "https://34a451b0-aa8b-4fa9-ba91-429e9ba4700f.us-west-2-0.aws.cloud.qdrant.io"

# 2. Your Key (Please rotate this later!)
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.oLxu6zd_DFnW4VNPXfSIw6BjaTn0dT0nQUjo3SBfkqo"

try:
    print(f"Attempting to connect to:\n{CLEAN_URL}...")
    client = QdrantClient(url=CLEAN_URL, api_key=API_KEY)
    
    # Try to list collections to prove we are in
    cols = client.get_collections()
    print("✅ SUCCESS! Connected to Qdrant Cloud.")
    print(f"Found collections: {cols}")

    # Test Gemini too
    print("\nTesting Gemini...")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key="AIzaSyCD4183FBL7R8N4uf1GZuVcNRuvHE0PS7A")
    res = llm.invoke("Hi")
    print(f"✅ Gemini Responded: {res.content}")

except Exception as e:
    print(f"\n❌ STILL FAILING: {e}")