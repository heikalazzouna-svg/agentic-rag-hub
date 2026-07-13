from pathlib import Path

import streamlit as st
import inngest
from dotenv import load_dotenv
from data_loader import embed_texts
from vector_db import QdrantStorage
import chat_db
import time
from ai_client import chat_completion, stream_chat_completion

load_dotenv()

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chat — Ask Your PDFs",
    page_icon="💬",
    layout="wide",
)

# ── Custom Styling ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Global ─────────────────────────────────────────────────── */
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif !important;
        background-color: #0b1120 !important;
        color: #e2e8f0 !important;
    }
    .stApp > header { background-color: transparent !important; }
    .block-container { padding-top: 1.5rem !important; }

    /* ── Sidebar ────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: #0d1526 !important;
        border-right: 1px solid #1e293b !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding: 1.2rem 1rem !important;
    }

    /* Sidebar section labels */
    .sidebar-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748b;
        margin: 1.2rem 0 0.5rem 0;
    }
    .sidebar-label:first-child { margin-top: 0; }

    /* App branding */
    .app-brand {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #1e293b;
    }
    .app-brand-icon {
        width: 36px; height: 36px;
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem;
    }
    .app-brand-text h3 {
        margin: 0; font-size: 1rem; font-weight: 700; color: #f1f5f9;
    }
    .app-brand-text p {
        margin: 0; font-size: 0.72rem; color: #64748b;
    }

    /* Upload drop zone */
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
        border: 1.5px dashed #1e3a5f !important;
        border-radius: 12px !important;
        background: rgba(15, 23, 42, 0.6) !important;
        padding: 0.5rem !important;
        transition: border-color 0.2s;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {
        border-color: #3b82f6 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] label {
        color: #94a3b8 !important; font-size: 0.8rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        background: transparent !important;
        color: #3b82f6 !important;
        border: none !important;
        font-size: 0.8rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] small {
        color: #475569 !important; font-size: 0.7rem !important;
    }

    /* Document library cards */
    .doc-card {
        background: #111c32;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 0.55rem 0.7rem;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        transition: background 0.2s;
    }
    .doc-card:hover { background: #162033; }
    .doc-icon {
        width: 30px; height: 30px;
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.8rem; flex-shrink: 0;
    }
    .doc-info { flex: 1; min-width: 0; }
    .doc-name {
        font-size: 0.78rem; font-weight: 500; color: #cbd5e1;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .doc-meta { font-size: 0.65rem; color: #475569; }

    /* Chunk badge */
    .chunk-badge {
        background: rgba(59, 130, 246, 0.12);
        color: #60a5fa;
        font-size: 0.65rem;
        font-weight: 600;
        padding: 0.15rem 0.5rem;
        border-radius: 20px;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }

    /* Settings slider */
    section[data-testid="stSidebar"] .stSlider label {
        color: #94a3b8 !important; font-size: 0.8rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stThumbValue"],
    section[data-testid="stSidebar"] [data-testid="stTickBarMin"],
    section[data-testid="stSidebar"] [data-testid="stTickBarMax"] {
        color: #60a5fa !important;
    }

    /* Clear chat button */
    section[data-testid="stSidebar"] .stButton button {
        background: transparent !important;
        border: 1px solid #1e293b !important;
        color: #94a3b8 !important;
        border-radius: 8px !important;
        font-size: 0.8rem !important;
        transition: all 0.2s !important;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        border-color: #ef4444 !important;
        color: #f87171 !important;
        background: rgba(239, 68, 68, 0.08) !important;
    }

    /* Sidebar form (URL input) */
    section[data-testid="stSidebar"] [data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
    }
    section[data-testid="stSidebar"] input {
        background: #111c32 !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        color: #cbd5e1 !important;
        font-size: 0.8rem !important;
    }
    section[data-testid="stSidebar"] input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        padding: 0.35rem 1rem !important;
    }

    /* ── Main Chat Area ──────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.6rem 0 !important;
    }
    /* Assistant messages get a subtle card */
    [data-testid="stChatMessage"][data-testid-role="assistant"] {
        background: #111c32 !important;
        border: 1px solid #1e293b !important;
        border-radius: 14px !important;
        padding: 1rem 1.2rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Chat input styling */
    [data-testid="stChatInput"] {
        border-color: #1e293b !important;
    }
    [data-testid="stChatInput"] textarea {
        background: #111c32 !important;
        border: 1px solid #1e293b !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-size: 0.9rem !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #3b82f6 !important;
    }
    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
        border: none !important;
        border-radius: 50% !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        font-size: 0.8rem !important;
        color: #64748b !important;
        background: transparent !important;
    }
    .streamlit-expanderContent {
        background: #0d1526 !important;
        border: 1px solid #1e293b !important;
        border-radius: 0 0 8px 8px !important;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 5rem 2rem;
    }
    .empty-state .emoji { font-size: 3rem; margin-bottom: 1rem; opacity: 0.5; }
    .empty-state h3 { color: #475569; font-weight: 600; font-size: 1.1rem; }
    .empty-state p { color: #334155; font-size: 0.85rem; }

    /* Footer text */
    .chat-footer {
        text-align: center;
        font-size: 0.7rem;
        color: #475569;
        padding: 0.5rem 0;
        position: fixed;
        bottom: 3.5rem;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        pointer-events: none;
    }

    /* Hide default streamlit elements */
    #MainMenu, footer { visibility: hidden; }

    /* Style the header bar but keep it visible for sidebar toggle */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Sidebar h2 override */
    section[data-testid="stSidebar"] .stMarkdown h2 {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# ── Gemini & Inngest Clients ────────────────────────────────────────────────


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)


# ── Session State Initialization ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = chat_db.get_messages()


# ── Helper Functions ────────────────────────────────────────────────────────
def save_uploaded_file(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path


def send_rag_ingest_event(file_path: Path) -> None:
    client = get_inngest_client()
    client.send_sync(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "file_path": str(file_path.resolve()),
                "pdf_path": str(file_path.resolve()),  # backward compat
                "source_id": file_path.name,
            },
        )
    )


def send_rag_ingest_url_event(url: str) -> None:
    client = get_inngest_client()
    client.send_sync(
        inngest.Event(
            name="rag/ingest_url",
            data={"url": url},
        )
    )


def build_conversation_context(messages: list, max_history: int = 5) -> str:
    """Build a conversation context string from the last N exchanges."""
    recent = messages[-(max_history * 2):]  # Each exchange = 2 messages (user + assistant)
    if not recent:
        return ""

    context_parts = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        context_parts.append(f"{role}: {msg['content']}")

    return "\n".join(context_parts)


def rag_search(question: str, top_k: int = 5) -> dict:
    """Step 1: Embed the question and search Qdrant for relevant chunks."""
    query_vec = embed_texts([question], input_type="query")[0]
    store = QdrantStorage()
    found = store.search(query_vec, top_k)
    return {"contexts": found["contexts"], "sources": found["sources"]}


def needs_rag(question: str, conversation_history: list) -> bool:
    """Agentic Router: Decide if the query needs document search."""
    conv_context = build_conversation_context(conversation_history, max_history=3)
    prompt = (
        "You are a routing agent for a document chat application. "
        "Your job is to decide if the user's latest message requires searching their uploaded documents, "
        "or if it is a general conversation/greeting that can be answered directly.\n\n"
        f"Recent conversation:\n{conv_context}\n\n"
        f"User's latest message: {question}\n\n"
        "Does this message require searching the user's uploaded documents? Answer exactly YES or NO."
    )
    try:
        response = chat_completion(
            prompt,
            system_instruction="Return only YES or NO.",
            temperature=0.0,
            max_output_tokens=5,
        )
        return "YES" in response.upper()
    except Exception:
        return True  # Default to RAG if the router fails


def rag_stream_answer(question: str, contexts: list, conversation_history: list = None, use_rag: bool = True):
    """Step 2: Stream the LLM answer word-by-word."""
    # Include conversation history for follow-up support
    conv_context = ""
    if conversation_history:
        conv_context = build_conversation_context(conversation_history)
        conv_context = f"\nPrevious conversation:\n{conv_context}\n\n"

    if use_rag:
        context_block = "\n\n".join(f"- {c}" for c in contexts)
        user_content = (
            "Use the following context retrieved from documents to answer the question.\n\n"
            f"Document context:\n{context_block}\n\n"
            f"{conv_context}"
            f"Question: {question}\n"
            "Answer concisely and helpfully using the document context above. "
            "If this is a follow-up question, use the conversation history for context."
        )
        system_instruction = (
            "You are a helpful assistant that answers questions using only the provided document context. "
            "Be concise but thorough. If the context doesn't contain enough information, say so honestly. "
            "Support follow-up questions by using the conversation history."
        )
    else:
        user_content = (
            f"{conv_context}"
            f"User: {question}\n"
        )
        system_instruction = (
            "You are a helpful and polite conversational AI assistant. "
            "Respond naturally to the user's input."
        )

    for chunk in stream_chat_completion(
        user_content,
        system_instruction=system_instruction,
        max_output_tokens=1024,
        temperature=0.2,
    ):
        yield chunk


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    # App branding
    st.markdown('''
    <div class="app-brand">
        <div class="app-brand-icon">🤖</div>
        <div class="app-brand-text">
            <h3>RAG Chat</h3>
            <p>Chat with your documents</p>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Upload section
    st.markdown('<div class="sidebar-label">Upload Documents</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop PDFs or browse",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=False,
    )

    if uploaded is not None:
        if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded.name:
            with st.spinner("⏳ Uploading and triggering ingestion..."):
                path = save_uploaded_file(uploaded)
                send_rag_ingest_event(path)
            st.session_state.last_uploaded = uploaded.name
            st.success(f"✅ Ingesting: **{path.name}**")
            time.sleep(1)
            st.rerun()

    # Web Page Ingestion
    st.markdown('<div class="sidebar-label">Web Page Ingestion</div>', unsafe_allow_html=True)
    
    with st.form(key="url_ingest_form", clear_on_submit=True):
        url_input = st.text_input("URL", placeholder="https://example.com/article", label_visibility="collapsed")
        submit_url = st.form_submit_button("Ingest URL", use_container_width=True)
        
        if submit_url and url_input:
            with st.spinner("⏳ Fetching and ingesting webpage..."):
                send_rag_ingest_url_event(url_input)
            st.success(f"✅ Ingesting: **{url_input}**")
            time.sleep(1)
            st.rerun()

    # Document Library
    try:
        db_sources = QdrantStorage().get_all_sources()
        total_chunks = sum(db_sources.values()) if db_sources else 0
        st.markdown(
            f'<div class="sidebar-label">Document Library '
            f'<span class="chunk-badge">{total_chunks} chunks</span></div>',
            unsafe_allow_html=True
        )
        
        if not db_sources:
            st.caption("No documents ingested yet.")
        else:
            for source, count in db_sources.items():
                # Determine icon based on source type
                icon = "🌐" if source.startswith("http") else "📄"
                ext = Path(source).suffix.lower() if not source.startswith("http") else "Web"
                
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(
                        f'<div class="doc-card">'
                        f'<div class="doc-icon">{icon}</div>'
                        f'<div class="doc-info">'
                        f'<div class="doc-name">{source}</div>'
                        f'<div class="doc-meta">{count} chunks · {ext.replace(".", "").upper() if ext != "Web" else ext}</div>'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )
                with col2:
                    if st.button("🗑️", key=f"del_{source}", help="Delete from AI memory"):
                        QdrantStorage().delete_source(source)
                        file_path = Path("uploads") / source
                        if file_path.exists():
                            file_path.unlink()
                        st.rerun()
    except Exception as e:
        st.error(f"Could not connect to Qdrant: {e}")

    # Settings
    st.markdown('<div class="sidebar-label">Settings</div>', unsafe_allow_html=True)
    top_k = st.slider(
        "Chunks to retrieve",
        min_value=1,
        max_value=10,
        value=4,
        step=1,
    )

    st.markdown("---")

    # Clear chat
    if st.button("🗑️  Clear chat", use_container_width=True):
        chat_db.clear_chat()
        st.session_state.messages = []
        st.rerun()


# ── Main Chat Area ──────────────────────────────────────────────────────────

# Render chat history
if not st.session_state.messages:
    st.markdown(
        '<div class="empty-state">'
        '<div class="emoji">💬</div>'
        "<h3>Ask anything about your documents</h3>"
        "<p>Upload a file or paste a URL in the sidebar to get started.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Sources used"):
                for src in msg["sources"]:
                    st.markdown(f"- `{src}`")

        # Show retrieved chunks for assistant messages
        if msg["role"] == "assistant" and msg.get("chunks"):
            with st.expander(f"🔍 Retrieved chunks ({len(msg['chunks'])})"):
                for i, chunk in enumerate(msg["chunks"], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.caption(chunk[:500] + ("..." if len(chunk) > 500 else ""))
                    if i < len(msg["chunks"]):
                        st.markdown("---")

# Chat input
if prompt := st.chat_input("Ask anything about your documents..."):
    # Add user message to DB and history
    chat_db.add_message(role="user", content=prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        sources = []
        chunks = []
        answer = ""

        try:
            # Smart Routing: Decide if we actually need to search documents
            with st.spinner("Thinking..."):
                require_search = needs_rag(prompt, st.session_state.messages[:-1])
                
            contexts = []
            
            if require_search:
                # Step 1: Search Qdrant (fast, show a brief spinner)
                with st.spinner("Searching documents..."):
                    search_result = rag_search(question=prompt, top_k=top_k)
                    contexts = search_result["contexts"]
                    sources = search_result["sources"]
                    chunks = contexts  # Save for display
            
            # Step 2: Stream the LLM answer word-by-word
            if require_search and not contexts:
                answer = "I couldn't find any relevant information in the uploaded documents. Try uploading a PDF first!"
                st.markdown(answer)
            else:
                stream = rag_stream_answer(
                    question=prompt,
                    contexts=contexts,
                    conversation_history=st.session_state.messages[:-1],
                    use_rag=require_search
                )
                answer = st.write_stream(stream)

            # Fire Inngest event for observability (non-blocking) only if it was a RAG query
            if require_search:
                try:
                    client = get_inngest_client()
                    client.send_sync(inngest.Event(
                        name="rag/query_pdf_ai",
                        data={"question": prompt, "top_k": top_k},
                    ))
                except Exception:
                    pass

        except Exception as e:
            answer = f"⚠️ Error: {e}"
            st.markdown(answer)
            sources = []
            chunks = []

        # Show sources
        if sources:
            with st.expander("📄 Sources used"):
                for src in sources:
                    st.markdown(f"- `{src}`")

        # Show retrieved chunks
        if chunks:
            with st.expander(f"🔍 Retrieved chunks ({len(chunks)})"):
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.caption(chunk[:500] + ("..." if len(chunk) > 500 else ""))
                    if i < len(chunks):
                        st.markdown("---")

    # Add assistant message to DB and history
    chat_db.add_message(
        role="assistant", 
        content=answer, 
        sources=sources, 
        chunks=chunks
    )
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "chunks": chunks,
    })

# Footer
st.markdown('<div class="chat-footer">Answers are grounded in your uploaded sources.</div>', unsafe_allow_html=True)
