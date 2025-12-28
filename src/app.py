import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Page Config
st.set_page_config(page_title="Oracle Sage", page_icon="🔮")

# 1. Load Secrets
load_dotenv()

def get_secret(key_name):
    value = os.getenv(key_name)
    if value:
        return value
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except:
        return None
    return None

GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")

# 2. Setup AI (Hybrid Mode)
# REMOVED st.toast from inside this function to fix the error
@st.cache_resource
def setup_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )
    
    local_path = "./qdrant_data"
    
    # We return the store AND a status message string
    if os.path.exists(local_path):
        client = QdrantClient(path=local_path)
        msg = "Local Database"
    else:
        qdrant_url = get_secret("QDRANT_URL")
        qdrant_key = get_secret("QDRANT_API_KEY")
        if not qdrant_url:
            raise ValueError("No Local DB found AND no Cloud URL provided.")
        client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        msg = "Cloud Database"
    
    vector_store = QdrantVectorStore(
        client=client, 
        collection_name="oracle_knowledge_base", 
        embedding=embeddings
    )
    return vector_store, msg

@st.cache_resource
def setup_chain(_vector_store):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )
    
    system_prompt = """
    You are 'Oracle Sage', an expert Oracle Consultant.
    Solve critical R12/Cloud issues using the context.
    If context is empty, use general expertise.
    
    Context:
    {context}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retriever = _vector_store.as_retriever(search_kwargs={"k": 3})
    return create_retrieval_chain(retriever, question_answer_chain)

# Initialize
try:
    if not GOOGLE_API_KEY:
        st.error("❌ Google API Key missing.")
        st.stop()
        
    # We unpack the tuple here
    vector_store, db_status = setup_vector_store()
    rag_chain = setup_chain(vector_store)
    
    # NOW it is safe to show the toast because we are outside the cached function
    st.toast(f"Connected to {db_status}", icon="✅")
    
except Exception as e:
    st.error(f"Connection Error: {e}")
    st.stop()

# --- UI LAYOUT ---
st.title("🔮 Oracle Sage Agent")
st.markdown("Your AI Techno-Functional Consultant")

# Sidebar: Training
with st.sidebar:
    st.header("📚 Knowledge Base")
    uploaded_file = st.file_uploader("Upload Oracle Manual (PDF)", type="pdf")
    
    if uploaded_file and st.button("Ingest to Brain"):
        with st.spinner("Reading..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = splitter.split_documents(docs)

                vector_store.add_documents(chunks)
                st.success(f"✅ Learned {len(chunks)} new chunks!")
                os.remove(tmp_path)
            except Exception as e:
                st.error(f"Failed: {e}")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about ORA errors, SQL, or Setup..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": prompt})
            answer = response['answer']
            st.markdown(answer)
            
    st.session_state.messages.append({"role": "assistant", "content": answer})