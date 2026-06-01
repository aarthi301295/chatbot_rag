from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


if __name__ == "__main__":
    embeddings = get_embeddings()
    vector = embeddings.embed_query(
        "What is Artificial Intelligence?"
    )
    print(len(vector))

    