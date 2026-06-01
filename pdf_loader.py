"""
PDF loading and chunking.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)


if __name__ == "__main__":
    docs = load_pdf("sample.pdf")
    print(f"Pages Loaded: {len(docs)}")
    chunks = split_documents(docs)
    print(f"Chunks Created: {len(chunks)}")

