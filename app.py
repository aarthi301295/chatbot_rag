"""
Main Streamlit application.
"""

import streamlit as st
from dotenv import load_dotenv

from pdf_loader import load_pdf, split_documents
from embeddings import get_embeddings
from retrieval import create_vector_store, retrieve_chunks
from llm import load_llm, generate_answer

load_dotenv()

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="AI Document Assistant",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------

st.markdown("""
<style>

/* Sidebar */

section[data-testid="stSidebar"]{
    background-color:#DCEEFF;
}

section[data-testid="stSidebar"] *{
    color:#0F172A !important;
}

/* Main Page */

.stApp{
    background-color:#F8FAFC;
}

/* Title */

.main-title{
    text-align:center;
    color:#0F172A;
    font-size:42px;
    font-weight:bold;
}

.sub-title{
    text-align:center;
    color:#475569;
    font-size:18px;
    margin-bottom:20px;
}

/* Answer Card */

.answer-card{
    background:white;
    padding:20px;
    border-radius:12px;
    border-left:6px solid #2563EB;
    box-shadow:0px 2px 10px rgba(0,0,0,0.08);
}

/* Center Layout */

.block-container{
    max-width:1200px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------

if "answer" not in st.session_state:
    st.session_state.answer = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

with st.sidebar:

    st.title("Document Assistant")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    st.markdown("---")

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.markdown(
    """
    <div class="main-title">
        AI Document Assistant
    </div>

    <div class="sub-title">
        Upload your PDF and ask questions
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# IMAGE
# -------------------------------------------------

st.image(
    "assets/background2.jpg",
    use_container_width=True
)

st.write("")

# -------------------------------------------------
# PROCESS PDF
# -------------------------------------------------

vector_store = None
documents = []
chunks = []

if uploaded_file:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing PDF..."):

        documents = load_pdf("temp.pdf")

        chunks = split_documents(
            documents
        )

        embeddings = get_embeddings()

        vector_store = create_vector_store(
            chunks,
            embeddings
        )

# -------------------------------------------------
# QUESTION INPUT
# -------------------------------------------------

st.markdown("### Ask a Question")

question = st.text_area(
    "",
    placeholder="Enter your question here...",
    height=90
)

col1, col2, col3 = st.columns([1, 1, 4])

with col1:
    ask_button = st.button(
        "Get Answer",
        use_container_width=False
    )

with col2:
    clear_button = st.button(
        "Clear",
        use_container_width=False
    )

# -------------------------------------------------
# CLEAR BUTTON
# -------------------------------------------------

if clear_button:

    st.session_state.answer = ""
    st.session_state.messages = []

    st.rerun()

# -------------------------------------------------
# CLEAR BUTTON
# -------------------------------------------------

if clear_button:

    st.session_state.answer = ""
    st.session_state.messages = []

    st.rerun()

# -------------------------------------------------
# GET ANSWER
# -------------------------------------------------

if ask_button:

    if not uploaded_file:

        st.error(
            "Please upload a PDF first."
        )

    elif not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        retrieved_docs = retrieve_chunks(
            vector_store,
            question
        )

        llm = load_llm()

        answer = generate_answer(
            llm,
            question,
            retrieved_docs
        )

        st.session_state.answer = answer

        st.session_state.messages.append(
            {
                "question": question,
                "answer": answer
            }
        )

# -------------------------------------------------
# ANSWER DISPLAY
# -------------------------------------------------

if st.session_state.answer:

    st.markdown("## Answer")

    st.markdown(
        f"""
        <div class="answer-card">
        {st.session_state.answer}
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------
# CHAT HISTORY
# -------------------------------------------------

if st.session_state.messages:

    st.markdown("## Previous Questions")

    for item in reversed(
        st.session_state.messages
    ):

        st.info(
            f"Question: {item['question']}"
        )

        st.success(
            f"Answer: {item['answer']}"
        )