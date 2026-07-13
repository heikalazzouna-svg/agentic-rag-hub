import csv
from pathlib import Path
from llama_index.readers.file import PDFReader, DocxReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from ai_client import embed_texts as nvidia_embed_texts

load_dotenv()

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv"}


def load_and_chunk_pdf(path: str):
    """Legacy function for backward compatibility."""
    return load_and_chunk_file(path)


def load_and_chunk_file(path: str):
    """Load and chunk a document. Supports PDF, DOCX, TXT, and CSV."""
    file_path = Path(path)
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        docs = PDFReader().load_data(file=path)
        texts = [d.text for d in docs if getattr(d, "text", None)]

    elif ext == ".docx":
        docs = DocxReader().load_data(file=path)
        texts = [d.text for d in docs if getattr(d, "text", None)]

    elif ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            texts = [f.read()]

    elif ext == ".csv":
        rows = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if headers:
                for row in reader:
                    # Convert each row to a readable string: "col1: val1, col2: val2, ..."
                    row_text = ", ".join(f"{h}: {v}" for h, v in zip(headers, row) if v.strip())
                    if row_text:
                        rows.append(row_text)
        texts = rows if rows else []

    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")

    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def load_and_chunk_url(url: str):
    """Scrape a webpage, extract text, and chunk it."""
    import requests
    from bs4 import BeautifulSoup
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.extract()
        
    text = soup.get_text(separator=" ", strip=True)
    
    chunks = splitter.split_text(text)
    return chunks



def embed_texts(texts: list[str], input_type: str = "passage") -> list[list[float]]:
    import time
    embeddings = []
    batch_size = 10  # Process in small batches to avoid rate limits
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings.extend(nvidia_embed_texts(batch, input_type=input_type))
        if i + batch_size < len(texts):
            time.sleep(1)  # Pause between batches to respect quota
            
    return embeddings
