"""
Sistem Tanya Jawab Dokumen PDF Berbasis Web Menggunakan RAG dengan LangChain.

Upload satu atau beberapa PDF, lalu ajukan beberapa pertanyaan tentang isinya.
Jawaban dihasilkan dengan Retrieval-Augmented Generation: teks PDF dipotong
menjadi chunk, di-embed, disimpan di vector store (Chroma), lalu chunk yang
paling relevan dengan pertanyaan diambil dan dikirim ke LLM sebagai konteks.
"""

import streamlit as st
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

st.set_page_config(page_title="Tanya Jawab Dokumen PDF (RAG)", page_icon="📄")
st.title("📄 Tanya Jawab Dokumen PDF (RAG)")
st.caption("Upload PDF, lalu ajukan pertanyaan tentang isinya.")

# ---------------------------------------------------------------------------
# Sidebar: pilih provider LLM + API key
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Konfigurasi")
    provider = st.selectbox("LLM Provider", ["Google Gemini", "OpenAI"])
    api_key = st.text_input(f"{provider} API Key", type="password")

    if provider == "Google Gemini":
        chat_model_name = st.text_input("Chat model", value="gemini-2.5-flash")
        embed_model_name = st.text_input("Embedding model", value="models/embedding-001")
        st.caption("Dapatkan API key gratis di aistudio.google.com/apikey")
    else:
        chat_model_name = st.text_input("Chat model", value="gpt-4o-mini")
        embed_model_name = st.text_input("Embedding model", value="text-embedding-3-small")
        st.caption("Dapatkan API key di platform.openai.com")

    k = st.slider("Jumlah chunk yang diambil (top-k)", min_value=2, max_value=8, value=4)


def get_llm_and_embeddings():
    """Buat instance LLM dan embedding model sesuai provider yang dipilih."""
    if provider == "Google Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

        llm = ChatGoogleGenerativeAI(model=chat_model_name, google_api_key=api_key, temperature=0.2)
        embeddings = GoogleGenerativeAIEmbeddings(model=embed_model_name, google_api_key=api_key)
    else:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        llm = ChatOpenAI(model=chat_model_name, api_key=api_key, temperature=0.2)
        embeddings = OpenAIEmbeddings(model=embed_model_name, api_key=api_key)
    return llm, embeddings


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Upload + proses PDF
# ---------------------------------------------------------------------------
uploaded_files = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)

process_clicked = st.button(
    "Proses Dokumen",
    disabled=not uploaded_files or not api_key,
    help="Upload PDF and isi API key terlebih dahulu" if (not uploaded_files or not api_key) else None,
)

if process_clicked:
    with st.spinner("Membaca dan mengindeks PDF..."):
        docs = []
        for f in uploaded_files:
            reader = PdfReader(f)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(
                        Document(page_content=text, metadata={"source": f.name, "page": i + 1})
                    )

        if not docs:
            st.error("Tidak ada teks yang bisa diekstrak dari PDF ini (mungkin hasil scan/gambar).")
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(docs)

            try:
                _, embeddings = get_llm_and_embeddings()
                st.session_state.vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
                st.session_state.messages = []
                st.success(f"{len(chunks)} potongan teks berhasil diindeks dari {len(uploaded_files)} PDF.")
            except Exception as e:
                st.error(f"Gagal membuat index: {e}")

st.divider()

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Tanyakan sesuatu tentang dokumen...")

if question:
    if st.session_state.vectorstore is None:
        st.warning("Upload dan proses PDF terlebih dahulu.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": k})
        relevant_docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in relevant_docs)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Kamu adalah asisten yang menjawab pertanyaan HANYA berdasarkan konteks "
                    "dokumen di bawah ini. Jika jawabannya tidak ada di dalam konteks, katakan "
                    "dengan jujur bahwa kamu tidak menemukan jawabannya di dokumen.\n\n"
                    "Konteks:\n{context}",
                ),
                ("human", "{question}"),
            ]
        )

        with st.chat_message("assistant"):
            with st.spinner("Berpikir..."):
                try:
                    llm, _ = get_llm_and_embeddings()
                    chain = prompt | llm
                    response = chain.invoke({"context": context, "question": question})
                    answer = response.content
                except Exception as e:
                    answer = f"Terjadi error saat memanggil LLM: {e}"
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.expander("Lihat sumber/chunk yang digunakan"):
            for d in relevant_docs:
                st.markdown(f"**{d.metadata.get('source')}** — halaman {d.metadata.get('page')}")
                st.text(d.page_content[:300] + "...")
