# Retrieval-Augmented Generation (RAG) Pipeline

An end-to-end **Retrieval-Augmented Generation (RAG)** pipeline built with **LangChain**, **Sentence-Transformers**, and **ChromaDB**.  
**Workflow:** 1. Ingestion — Load PDF documents from a folder; 2. Chunking — Split documents into overlapping text chunks; 3. Embedding — Convert chunks into dense vector embeddings; 4. Storage — Persist embeddings in a Chroma vector store; 5. Retrieval — Perform semantic search over the vector store given a natural-language query.

> An end-to-end Retrieval-Augmented Generation (RAG) pipeline built with LangChain, Sentence-Transformers, and ChromaDB.  
> Workflow: 1. Ingestion - Load PDF documents from a folder.

---

## Overview

This repository contains a production-oriented Python implementation of a RAG pipeline that demonstrates how to:

- Ingest documents (PDF / text) into LangChain `Document` objects.
- Chunk long documents into overlapping segments suitable for embedding.
- Generate dense vector embeddings using **Sentence-Transformers** and optionally **Groq** model embeddings where applicable.
- Persist embeddings and metadata in a **ChromaDB** vector store for efficient semantic retrieval.
- Execute semantic search and return ranked, scored document chunks with metadata.

This implementation is modular and designed to be readable and recruiter-friendly, suitable for showcasing on GitHub or a resume.

---

## Key Features

- **Modular design**: clear separation of ingestion, chunking, embedding, storage, and retrieval.
- **Multiple embedding backends**: primary use of `sentence-transformers` (`all-MiniLM-L6-v2`) and optional integration with **Groq** model embeddings for low-latency, hardware-accelerated inference where available.
- **Persistent vector store**: ChromaDB-backed collection for reproducible experiments and scalable retrieval.
- **Professional structure**: imports consolidated, docstrings preserved, concise comments, and top-to-bottom runnable script.

---

## Architecture Diagram

```mermaid
flowchart TD
    A[📂 Documents (PDF / TXT)] --> B[🔹 Ingestion\nLoad into LangChain Document objects]
    B --> C[✂️ Chunking\nSplit into overlapping text chunks]
    C --> D[🧠 Embedding\nSentence-Transformers or Groq model]
    D --> E[💾 Storage\nChromaDB persistent vector store]
    E --> F[🔍 Retrieval\nSemantic search with natural-language query]
    F --> G[📜 Results\nRanked chunks with metadata]

pip install langchain langchain-core langchain-community \
    langchain-text-splitters pypdf pymupdf sentence-transformers \
    chromadb scikit-learn
