# RAG Chat — Ask Your Documents 💬

A modern, production-ready Retrieval-Augmented Generation (RAG) application that allows you to upload documents or paste URLs and chat with them in real-time. Built with a sleek dark-mode UI, it features intelligent routing, background job processing, and high-performance vector search.

<!-- PLACE IMAGE 1 HERE (Main Interface Overview) -->
![App Interface - Dark Mode UI](assets/image1_interface.png)

## ✨ Features
* **Modern UI**: A beautiful, dark-themed Streamlit interface built for excellent user experience.
* **Multi-Format Ingestion**: Upload PDFs, Word documents (.docx), text files (.txt), and CSVs, or ingest entire web pages via URL.
* **NVIDIA AI Powered**: Uses state-of-the-art NVIDIA endpoints (`meta/llama-3.1-70b-instruct` for chat and `nvidia/nv-embedqa-e5-v5` for embeddings).
* **Agentic Routing**: Smartly determines if a question requires searching the vector database or if it can be answered natively, saving time and tokens.
* **Background Processing**: Powered by [Inngest](https://www.inngest.com/) for reliable, asynchronous document chunking and embedding.
* **Vector Storage**: Uses [Qdrant](https://qdrant.tech/) locally via Docker for blazing-fast semantic search.
* **Persistent Memory**: Chat history is automatically saved to a local SQLite database so you never lose a conversation.

<!-- PLACE IMAGE 2 HERE (Execution / Chat Example) -->
![App Execution - Example Chat and Retrieval](assets/image2_execution.png)

## 🏗️ Architecture Stack
* **Frontend**: Streamlit
* **Backend**: FastAPI + Uvicorn
* **Task Queue**: Inngest
* **Vector DB**: Qdrant (Docker)
* **LLM & Embeddings**: NVIDIA API (OpenAI Compatible)
* **Orchestration**: LlamaIndex (for chunking & parsing)

## 🚀 Getting Started

### Prerequisites
* Python 3.12+
* [uv](https://github.com/astral-sh/uv) (for lightning-fast dependency management)
* Docker Desktop (for Qdrant)
* Node.js / npx (for the Inngest Dev Server)
* An NVIDIA API Key

### 1. Environment Variables
Create a `.env` file in the root directory and add your keys:
```env
NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_CHAT_MODEL=meta/llama-3.1-70b-instruct
NVIDIA_EMBED_MODEL=nvidia/nv-embedqa-e5-v5
```

### 2. Start the Services
You will need to open **four separate terminals** to run the complete stack locally.

**Terminal 1: Start Qdrant (Vector Database)**
```bash
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage:z qdrant/qdrant
```

**Terminal 2: Start Inngest (Background Jobs)**
```bash
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest
```

**Terminal 3: Start FastAPI (Backend Engine)**
```bash
uv run python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

**Terminal 4: Start Streamlit (Frontend UI)**
```bash
uv run python -m streamlit run streamlit_app.py
```

Once all services are running, the application will automatically open in your browser at `http://localhost:8501`.

## 📂 Project Structure
* `streamlit_app.py`: The frontend UI, chat loop, and design system.
* `main.py`: The FastAPI server and Inngest background event handlers.
* `data_loader.py`: Handles file parsing, web scraping, chunking, and embedding texts.
* `vector_db.py`: Manages the connection and operations with the Qdrant database.
* `ai_client.py`: Wrapper for the NVIDIA (OpenAI-compatible) API client.
* `chat_db.py`: Manages the SQLite database for persistent chat history.
