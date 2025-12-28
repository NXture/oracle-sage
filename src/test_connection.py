import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Load the secrets
load_dotenv()

def run_diagnostics():
    print("--- 🩺 Oracle Sage Diagnostics (Gemini Edition) ---")
    
    # Check 1: Can we find the keys?
    google_key = os.getenv("GOOGLE_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    
    if not google_key:
        print("❌ CRITICAL: GOOGLE_API_KEY missing in .env")
        return
    if not qdrant_url:
        print("❌ CRITICAL: Qdrant URL missing in .env")
        return
        
    print("✅ Environment Variables Loaded.")

    # Check 2: Can we talk to the Brain (Gemini)?
    try:
        # We use 'gemini-1.5-flash' because it's fast and free
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        response = llm.invoke("Hello! Are you ready to be an Oracle expert?")
        
        print(f"✅ Gemini Connection Successful! The Agent says:\n   \"{response.content}\"")
    except Exception as e:
        print(f"❌ Gemini Connection Failed: {e}")

    # Check 3: Can we talk to the Memory (Qdrant)?
    try:
        q_client = QdrantClient(
            url=qdrant_url,
            api_key=os.getenv("QDRANT_API_KEY")
        )
        collections = q_client.get_collections()
        print(f"✅ Qdrant Connection Successful! Found {len(collections.collections)} existing memory collections.")
    except Exception as e:
        print(f"❌ Qdrant Connection Failed: {e}")

if __name__ == "__main__":
    run_diagnostics()