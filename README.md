# 📚 Retrieval-Augmented Generation (RAG) Pipeline

An end-to-end **Retrieval-Augmented Generation (RAG)** pipeline built with **LangChain**, **Sentence-Transformers**, and **ChromaDB**.  
This project demonstrates how to ingest documents, embed them into dense vector representations, store them in a vector database, and perform semantic search to retrieve relevant information.

---

## 🚀 Workflow Overview

1. **Ingestion**  
   Load PDF documents from a folder into LangChain `Document` objects.

2. **Chunking**  
   Split documents into overlapping text chunks using a recursive character-based splitter.

3. **Embedding**  
   Convert text chunks into dense vector embeddings with `Sentence-Transformers`.

4. **Storage**  
   Persist embeddings in a **ChromaDB vector store** for efficient retrieval.

5. **Retrieval**  
   Perform semantic search over the vector store given a natural-language query.

---

## 🛠️ Dependencies

Install the required packages:

```bash
pip install langchain langchain-core langchain-community \
    langchain-text-splitters pypdf pymupdf sentence-transformers \
    chromadb scikit-learn
