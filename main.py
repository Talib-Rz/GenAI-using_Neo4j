import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


class Neo4jConnection:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def run_query(self, query, parameters=None):
        with self._driver.session() as session:
            result = session.run(query, parameters or {})
            return list(result)  

# Initialize Neo4j connection
neo4j_conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)


st.title("PDF Chunk Storage & Retrieval with Neo4j")


uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    pdf_path = f"./temp/{uploaded_file.name}"
    os.makedirs("./temp", exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    st.write(f"Uploaded file: **{uploaded_file.name}**")


    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)


    for i, chunk in enumerate(chunks):
        query = """
        CREATE (c:Chunk {id: $id, text: $text})
        """
        neo4j_conn.run_query(query, {"id": i, "text": chunk.page_content})

    st.success(f"Stored {len(chunks)} chunks in Neo4j!")

# Search Query Input
query_text = st.text_input("Enter a search query:")

if query_text:
    query = """
    MATCH (c:Chunk)
    WHERE c.text CONTAINS $query_text
    RETURN c.text LIMIT 5
    """
    results = neo4j_conn.run_query(query, {"query_text": query_text})

    st.subheader("Retrieved Chunks:")
    if results:
        for record in results:
            st.write(record["c.text"])
            st.write("---------------------------")
    else:
        st.write("No matching chunks found.")


neo4j_conn.close()
