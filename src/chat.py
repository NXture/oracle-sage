import os
import textwrap
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Setup Brain
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# 2. Setup Memory (LOCAL)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

print(f"🔌 Connecting to Local Database...")
# Connect to the folder on your disk
client = QdrantClient(path="./qdrant_data") 

vector_store = QdrantVectorStore(
    client=client, 
    collection_name="oracle_knowledge_base", 
    embedding=embeddings
)

# 3. Persona
system_prompt = """
You are 'Oracle Sage', a world-class Oracle Techno-Functional Consultant.
Your Goal: Solve critical Oracle R12/Cloud issues using the provided context.

Process:
1. ANALYZE the error code or symptom.
2. CHECK the provided context for official fixes.
3. If context is empty, use your general Oracle knowledge to debug directly.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 4. Build Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 5. Start Chat
def start_chat():
    print("\n🔮 Oracle Sage is ONLINE (Local Mode). (Type 'exit' to quit)")
    print("--------------------------------------------------")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        print("🤔 Thinking...", end="\r")
        try:
            response = rag_chain.invoke({"input": query})
            print(f"\nSage: {textwrap.fill(response['answer'], width=80)}")
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    start_chat()