import streamlit as st
import os
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from neo4j import GraphDatabase
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings 
from sklearn.metrics.pairwise import cosine_similarity 

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def run_query(self, query, parameters=None):
        with self._driver.session() as session:
            result = session.run(query, parameters or {})
            return [record for record in result]

neo4j_conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

st.title("PDF Chunk & Embedding Storage with Neo4j")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    pdf_path = f"./temp/{uploaded_file.name}"
    os.makedirs("./temp", exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"Uploaded file: **{uploaded_file.name}**")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk_text = chunk.page_content
        embedding_vector = embed_model.embed_documents([chunk_text])[0]  

        query = """
        MERGE (c:Chunk {id: $id, text: $text})
        MERGE (e:Embedding {id: $id, vector: $vector})
        MERGE (c)-[:STORES_IN]->(e)
        """
        neo4j_conn.run_query(query, {"id": i, "text": chunk_text, "vector": embedding_vector})

    st.success(f"Stored {len(chunks)} chunks and embeddings in Neo4j!")

query_text = st.text_input("🔍 Enter a search query:")

if query_text:
    query_embedding = embed_model.embed_query(query_text)  

    query = """
    MATCH (c:Chunk)-[:STORES_IN]->(e:Embedding)
    RETURN c.text AS text, e.vector AS vector
    """
    results = neo4j_conn.run_query(query)

    if results:
        stored_texts = [record['text'] for record in results]
        stored_vectors = np.array([record['vector'] for record in results]) 

        query_vector = np.array(query_embedding).reshape(1, -1)
        similarities = cosine_similarity(query_vector, stored_vectors)[0] 

        sorted_indices = np.argsort(similarities)[::-1] 
        top_k = 5 

        st.subheader("Retrieved Chunks:")
        for idx in sorted_indices[:top_k]:
            st.write(f"**Text:** {stored_texts[idx]}")
            st.write(f"**Similarity Score:** {similarities[idx]:.4f}")
            st.write("------------------------------------------------")
    else:
        st.write("No relevant chunks found.")

# # Close Neo4j connection
# neo4j_conn.close()
