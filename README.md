# Rust PDF Chatbot with RAG

An AI-powered PDF question-answering application built using **Rust, Axum, and Google Gemini API**.

Upload a PDF document and ask questions about its contents. The application extracts the text, divides it into smaller chunks, retrieves relevant sections using TF-IDF and cosine similarity, and uses Google Gemini to generate answers based on the retrieved context.

## Features

- Upload PDF documents (up to 20 MB).
- Extract text from PDF files.
- Split documents into overlapping text chunks.
- Build a TF-IDF-based document index.
- Retrieve relevant sections using cosine similarity.
- Select the top three relevant chunks.
- Generate answers using Google Gemini API.
- REST API backend using Axum.
- Asynchronous processing using Tokio.
- Shared application state using Arc and RwLock.
- Input validation and error handling.

## Tech Stack

| Technology | Purpose |
|---|---|
| Rust | Backend programming language |
| Axum | REST API framework |
| Tokio | Asynchronous runtime |
| Reqwest | HTTP requests to Gemini |
| Serde | JSON serialization and deserialization |
| pdf-extract | PDF text extraction |
| TF-IDF | Text relevance calculation |
| Cosine Similarity | Relevant document retrieval |
| Google Gemini API | AI answer generation |
| HTML, CSS, JavaScript | Frontend |

## Project Structure

```text
rust-pdf-chat/
│
├── frontend/
│   └── index.html
│
├── src/
│   ├── main.rs
│   └── rag.rs
│
├── Cargo.toml
├── Cargo.lock
├── .env
├── .gitignore
└── README.md
```

## How It Works

### 1. PDF Upload
The user uploads a PDF through the web interface. The Axum backend validates the file and extracts readable text using `pdf-extract`.

### 2. Text Chunking
The extracted text is divided into smaller sections.

- Chunk size: 1,200 characters.
- Overlap: 200 characters.

### 3. TF-IDF Indexing
The application calculates Term Frequency (TF) and Inverse Document Frequency (IDF) to create normalized text vectors for each chunk.

### 4. Relevant Chunk Retrieval
When a user asks a question:
1. The question is tokenized.
2. A TF-IDF vector is generated.
3. Cosine similarity is calculated against the document chunks.
4. The top three matching chunks are selected.

### 5. AI Answer Generation
The retrieved document sections and user's question are sent to Google Gemini. Gemini generates an answer using the supplied context.

## Requirements

- Rust and Cargo.
- Google Gemini API key.
- Internet connection.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Vipu-l/rust-pdf-chat.git
cd rust-pdf-chat
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-flash-lite-latest
```

Replace the placeholder with your own Gemini API key.

**Important:** Never upload your actual `.env` file or API key to GitHub.

### 3. Build the application

```bash
cargo build
```

### 4. Run the application

```bash
cargo run
```

### 5. Open the application

Visit:

http://127.0.0.1:3000

## API Endpoints

### Health Check

`GET /api/health`

Checks whether the backend is running.

### Upload PDF

`POST /api/upload`

Uploads and processes a PDF document.

Request type: `multipart/form-data`

Field name: `file`

Maximum file size: 20 MB.

### Chat

`POST /api/chat`

Asks a question about the uploaded PDF.

Example request:

```json
{
  "question": "What is this document about?"
}
```

## Current Limitations

- Only one PDF is stored at a time.
- Documents are stored in application memory.
- Scanned PDFs require OCR, which is not currently implemented.
- Retrieval uses TF-IDF-based lexical matching, not neural semantic embeddings.
- When no matching keywords are found, the system falls back to the first document chunks.
- Google Gemini API access requires an internet connection and a valid API key.

## Future Improvements

- Semantic search using embedding models.
- Multiple PDF document support.
- OCR for scanned documents.
- Persistent vector storage.
- Conversation history.
- Streaming AI responses.
- User authentication.

## Author

Vipul Pandey

GitHub: https://github.com/Vipu-l