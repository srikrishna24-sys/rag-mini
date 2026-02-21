
# 📚 rag-mini

A simple Retrieval-Augmented Generation (RAG) application built for learning and upskilling purposes.

This project demonstrates how to:

* Ingest documents
* Create embeddings
* Store vectors in FAISS
* Retrieve relevant context
* Generate grounded answers using an LLM
* Build a clean CLI interface

---

## 🏗 Project Architecture

```
rag-mini/
│
├── settings.py
├── ingest.py
├── retriever.py
├── generate.py
├── cli.py
│
├── data/
├── indexes/
└── .env
```

---

## ⚙️ settings.py — Configuration Layer

Centralized configuration management.

**Responsibilities:**

* Loads environment variables from `.env`
* Provides project configuration such as:

  * `OPENAI_API_KEY`
  * Embedding model name
  * Index folder path
  * Chunk size
  * Chunk overlap

**Goal:**
Keep all configuration in one place to make the system clean and maintainable.

---

## 📥 ingest.py — Build the Library Index

Run this command:

```bash
rag-mini ingest
```

### What it does:

1. Reads all `.md` and `.txt` files from the `data/` folder
2. Splits documents into small chunks
3. Converts each chunk into vector embeddings using OpenAI
4. Stores embeddings in FAISS
5. Saves metadata (chunk text + source file info)

### Output:

```
indexes/faiss.index        # Vector index
indexes/metadata.jsonl     # Chunk text + source information
```

---

## 🔎 retriever.py — Retrieve Relevant Chunks

Triggered when running:

```bash
rag-mini ask "your question"
```

### What it does:

1. Converts your question into an embedding
2. Searches FAISS for closest vector matches
3. Loads metadata to retrieve:

   * Chunk text
   * Source file path
   * Similarity score

### Output:

* Top-K most relevant chunks
* Similarity scores

---

## 🤖 generate.py — Generate Grounded Answer

This is the LLM layer.

### What it does:

1. Takes retrieved chunks
2. Builds a structured **CONTEXT block with citations**
3. Calls the OpenAI Responses API
4. Forces the model to answer only from retrieved context

### Output:

* Clean final answer
* Source citations

---

## 🖥 cli.py — Command Line Interface

Acts as the glue between all modules.

### Available Commands:

#### 🔹 Ingest Documents

```bash
rag-mini ingest
```

Builds the FAISS index from files inside `data/`.

#### 🔹 Ask a Question

```bash
rag-mini ask "What is RAG?"
```

Retrieves relevant chunks and generates a grounded answer.

---

## 🧠 Learning Goals

This project helps you understand:

* How embeddings work
* How vector databases (FAISS) function
* The separation between retrieval and generation
* How to structure a production-style RAG system
* Clean software architecture for AI applications

---

## 📦 Requirements

* Python 3.9+
* FAISS
* OpenAI SDK
* python-dotenv

---

## 🔐 Environment Setup

Create a `.env` file:

```env
OPENAI_API_KEY=your_key_here
```

---

## 🚀 Future Improvements

* Add streaming responses
* Add document upload support
* Add reranking layer
* Add evaluation metrics
* Add web UI (FastAPI / Streamlit)

---

If you want, I can now:

* 🔥 Make it look like a serious production-grade open source repo
* 📈 Add badges + architecture diagram
* 🧠 Add explanation section for interviews
* 🏗 Help you push this cleanly to GitHub with proper commits
* 📄 Generate a LICENSE + CONTRIBUTING file

Tell me your goal — learning project or portfolio project?
