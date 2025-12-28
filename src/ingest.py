import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()
print("--- 🚀 Starting LOCAL Knowledge Ingestion ---")

# 1. Load Data
loader = TextLoader("./data/oracle_knowledge.txt")
documents = loader.load()

# 2. Chunk Data
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"✅ Split into {len(chunks)} chunks.")

# 3. Setup Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# 4. SAVE TO LOCAL DISK (The Fix)
# Instead of a URL, we give it a path folder.
print("⏳ Saving to local file database...")
try:
    QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        path="./qdrant_data",  # <--- SAVES LOCALLY
        collection_name="oracle_knowledge_base",
        force_recreate=True
    )
    print("✅ SUCCESS! Knowledge saved to './qdrant_data' folder.")
except Exception as e:
    print(f"❌ Ingest Failed: {e}")