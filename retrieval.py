"""
FAISS vector store operations.
"""

from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embeddings):
    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )
    return vector_store


def retrieve_chunks(vector_store, question):
    docs = vector_store.similarity_search(
        question,
        k=4
    )
    return docs

if __name__ == "__main__":
    print("retrieval.py loaded successfully")