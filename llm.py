"""
Groq response generation.
"""

from langchain_groq import ChatGroq


def load_llm():

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3
    )

    return llm


def generate_answer(llm, question, retrieved_docs):

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    prompt = f"""
You are a helpful AI assistant.

Use the provided context as your source of truth.

Answer the user's question in a natural,
clear and professional way.

Do not copy the context word-for-word
unless necessary.

Explain the answer in your own words
while staying faithful to the information
contained in the document.

Context:
{context}

Question:
{question}

If the answer cannot be found in the context,
respond:

"I could not find that information in the document."
"""

    response = llm.invoke(prompt)

    return response.content