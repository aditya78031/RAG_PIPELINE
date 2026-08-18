<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>RAG Pipeline — README</title>
  <style>
    :root{
      --bg:#0f1720; --card:#0b1220; --muted:#9aa4b2; --accent:#7dd3fc;
      --text:#e6eef6; --mono:#cbd5e1;
    }
    body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial; background:linear-gradient(180deg,#071021 0%, #071827 100%); color:var(--text); margin:0; padding:32px;}
    .container{max-width:980px; margin:0 auto; background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:12px; padding:28px; box-shadow:0 6px 30px rgba(2,6,23,0.6);}
    h1{font-size:28px; margin:0 0 6px 0;}
    .tagline{color:var(--muted); margin-bottom:18px;}
    hr{border:none; height:1px; background:linear-gradient(90deg, transparent, rgba(125,211,252,0.12), transparent); margin:22px 0;}
    h2{font-size:18px; margin:18px 0 8px 0; color:var(--accent);}
    p{color:var(--muted); line-height:1.6; margin:8px 0;}
    ul{color:var(--muted); margin:8px 0 8px 20px;}
    pre{background:#071827; border:1px solid rgba(125,211,252,0.06); padding:14px; border-radius:8px; overflow:auto; color:var(--mono); font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, "Roboto Mono", monospace; font-size:13px;}
    code{background:rgba(125,211,252,0.03); padding:2px 6px; border-radius:6px; color:var(--accent);}
    .grid{display:grid; grid-template-columns:1fr; gap:14px;}
    .note{color:#bfe9ff; background:linear-gradient(180deg, rgba(125,211,252,0.02), rgba(125,211,252,0.01)); padding:10px; border-radius:8px; border:1px solid rgba(125,211,252,0.04);}
    .muted{color:var(--muted);}
    .mermaid-wrapper{background:#071827; padding:12px; border-radius:8px; border:1px solid rgba(125,211,252,0.04);}
    footer{margin-top:22px; color:var(--muted); font-size:13px;}
    .topics{margin-top:8px; color:var(--muted);}
    .small{font-size:13px; color:var(--muted);}
  </style>
  <!-- Mermaid CDN for rendering the workflow diagram -->
  <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
  <script>mermaid.initialize({startOnLoad:true, theme: 'dark'});</script>
</head>
<body>
  <div class="container">
    <header>
      <h1>Retrieval-Augmented Generation (RAG) Pipeline</h1>
      <div class="tagline">A production‑oriented RAG pipeline for document ingestion, embedding, storage, and semantic retrieval.</div>
    </header>

    <section class="grid">
      <div>
        <h2>What this repo contains</h2>
        <p>This repository implements a modular, end‑to‑end Retrieval‑Augmented Generation (RAG) pipeline in Python. It demonstrates document ingestion (PDF/TXT), chunking, embedding (Sentence‑Transformers with optional Groq backend), persistent storage in ChromaDB, and semantic retrieval via a simple retriever interface.</p>
      </div>

      <hr />

      <div>
        <h2>Architecture / Workflow</h2>
        <div class="mermaid-wrapper">
          <div class="mermaid">
flowchart TD
  A[📂 Documents (PDF / TXT)] --> B[🔹 Ingestion\n(load_all_pdfs / TextLoader)]
  B --> C[✂️ Chunking\n(split_docs)]
  C --> D[🧠 Embedding\n(EmbeddingManager: Sentence-Transformers or Groq)]
  D --> E[💾 Storage\n(VectorStoreManager → ChromaDB)]
  E --> F[🔍 Retrieval\n(RAGRetriever)]
  F --> G[📜 Results\nRanked chunks with metadata]
          </div>
        </div>
        <p class="small muted">Diagram rendered with Mermaid. The pipeline maps documents → ingestion → chunking → embedding → storage → retrieval → results.</p>
      </div>

      <hr />

      <div>
        <h2>Key features</h2>
        <ul>
          <li><strong>Modular design</strong>: clear separation of ingestion, chunking, embedding, storage, and retrieval.</li>
          <li><strong>Multiple embedding backends</strong>: default <code>Sentence-Transformers</code> (<code>all-MiniLM-L6-v2</code>) with an option to integrate <strong>Groq</strong> for hardware‑accelerated inference.</li>
          <li><strong>Persistent vector store</strong>: ChromaDB collection for reproducible experiments and scalable retrieval.</li>
          <li><strong>Recruiter‑friendly</strong>: concise docstrings, clear configuration, and runnable examples for quick evaluation.</li>
        </ul>
      </div>

      <hr />

      <div>
        <h2>Dependencies & installation</h2>
        <p class="muted">Install core dependencies for the pipeline:</p>
        <pre><code>pip install langchain langchain-core langchain-community \
    langchain-text-splitters pypdf pymupdf sentence-transformers \
    chromadb scikit-learn</code></pre>
        <p class="muted"><strong>Groq integration note:</strong> Groq requires provider-specific SDKs, drivers, and runtime components. Follow Groq’s official installation and runtime instructions for model artifacts and hardware setup before enabling Groq as an embedding backend.</p>
      </div>

      <hr />

      <div>
        <h2>Configuration</h2>
        <p class="muted">Top-level variables to edit in the script:</p>
        <pre><code>TEXT_DATA_PATH = "data/Python.txt"
PDF_FOLDER_PATH = "data/pdfs"
VECTOR_STORE_DIR = "data/vector_store"
VECTOR_STORE_COLLECTION = "pdf_documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"</code></pre>
        <p class="muted">Adjust these paths and the <code>EMBEDDING_MODEL_NAME</code> to match your environment and preferred embedding model.</p>
      </div>

      <hr />

      <div>
        <h2>Usage</h2>
        <p class="muted">Run the pipeline end-to-end. The example below uses the same function and class names present in the repository: <code>EmbeddingManager</code>, <code>VectorStoreManager</code>, <code>RAGRetriever</code>, <code>split_docs</code>, <code>load_all_pdfs</code>.</p>

        <pre><code>from your_module import (
    demonstrate_document_basics,
    demonstrate_text_loading,
    load_all_pdfs,
    split_docs,
    EmbeddingManager,
    VectorStoreManager,
    RAGRetriever,
)

def run_pipeline():
    demonstrate_document_basics()
    demonstrate_text_loading()

    # Ingest PDFs and split into chunks
    all_pdf_documents = load_all_pdfs()
    chunks = split_docs(all_pdf_documents)

    # Initialize embedding and vector store managers
    embedding_manager = EmbeddingManager()          # Sentence-Transformers by default
    vector_store = VectorStoreManager()             # ChromaDB persistent store

    # Generate embeddings and persist
    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)
    vector_store.add_documents(chunks, embeddings)

    # Query the store
    rag_retriever = RAGRetriever(embedding_manager, vector_store)
    results = rag_retriever.retrieve("What is encoder decoder")

    for r in results:
        print(r["similarity_score"], r["metadata"], r["document"][:200])

if __name__ == "__main__":
    run_pipeline()</code></pre>

        <p class="muted">Command line:</p>
        <pre><code>python rag_pipeline.py</code></pre>
      </div>

      <hr />

      <div>
        <h2>Extending for Groq embeddings</h2>
        <p class="muted">To add Groq as an alternative embedding backend:</p>
        <ol class="muted">
          <li>Abstract the embedding interface: ensure <code>EmbeddingManager</code> exposes <code>generate_embeddings(text)</code> and can be subclassed.</li>
          <li>Implement <code>GroqEmbeddingManager</code> that calls Groq SDK/runtime and returns embeddings matching the shape of <code>SentenceTransformer.encode()</code>.</li>
          <li>Switch at runtime via a config flag or environment variable to instantiate either <code>EmbeddingManager</code> or <code>GroqEmbeddingManager</code> without changing downstream code.</li>
        </ol>

        <pre><code>class GroqEmbeddingManager(EmbeddingManager):
    def __init__(self, model_name, groq_client, **kwargs):
        self.model_name = model_name
        self.groq_client = groq_client
        # initialize Groq runtime / model artifact

    def generate_embeddings(self, text):
        # call Groq inference endpoint / runtime
        # return numpy array or list of vectors matching expected shape
        pass</code></pre>
      </div>

      <hr />

      <div>
        <h2>Best practices & notes</h2>
        <ul>
          <li><strong>Metadata hygiene</strong>: include <code>source</code>, <code>doc_index</code>, <code>page</code>, and <code>content_length</code> for actionable retrieval results.</li>
          <li><strong>Persist vector store</strong>: keep ChromaDB persisted to avoid re-embedding large corpora.</li>
          <li><strong>Embedding dimensionality</strong>: verify vector size when switching models and adapt storage/schema if needed.</li>
          <li><strong>Security</strong>: sanitize uploaded documents and avoid ingesting sensitive data in public repos.</li>
          <li><strong>Reproducibility</strong>: pin model versions and record environment details (Python version, package versions).</li>
        </ul>
      </div>

      <hr />

      <div>
        <h2>Evaluation & validation</h2>
        <ul>
          <li><strong>Sanity checks</strong>: assert <code>len(chunks) &gt; 0</code> and verify <code>embeddings.shape</code> after encoding.</li>
          <li><strong>Retrieval quality</strong>: run queries with known answers and confirm top‑k contains expected chunks.</li>
          <li><strong>Performance benchmarking</strong>: measure throughput (texts/sec) for Sentence‑Transformers vs Groq on your hardware.</li>
          <li><strong>Unit tests</strong>: add tests for <code>split_docs</code>, <code>EmbeddingManager.generate_embeddings</code>, and <code>VectorStoreManager.add_documents</code>.</li>
        </ul>
      </div>

      <hr />

      <div>
        <h2>Example outputs</h2>
        <p class="muted">Sample retrieval result (JSON-like):</p>
        <pre><code>[
  {
    "id": "doc_3f2a1b4e-...",
    "similarity_score": 0.92,
    "metadata": {"source": "data/pdfs/example.pdf", "doc_index": 12, "content_length": 1024},
    "document": "An encoder–decoder architecture consists of an encoder that..."
  },
  {
    "id": "doc_7c9d2a1f-...",
    "similarity_score": 0.87,
    "metadata": {"source": "data/pdfs/another.pdf", "doc_index": 3, "content_length": 980},
    "document": "In sequence-to-sequence models, the decoder generates..."
  }
]</code></pre>
      </div>

      <hr />

      <div>
        <h2>Contributing</h2>
        <ul>
          <li>Implement new embedding backends by subclassing <code>EmbeddingManager</code>.</li>
          <li>Add unit tests for chunking, embedding, and retrieval.</li>
          <li>Improve documentation with sample datasets and evaluation notebooks (keep notebooks separate from production <code>.py</code> scripts).</li>
        </ul>
      </div>

      <hr />

      <div>
        <h2>License</h2>
        <p class="muted"><strong>MIT</strong> (placeholder) — replace with your preferred license in <code>LICENSE</code>.</p>
      </div>

      <hr />

      <div>
        <h2>Contact</h2>
        <p class="muted">GitHub: <a href="https://github.com/<your-username>" style="color:var(--accent)">https://github.com/&lt;your-username&gt;</a><br />
        LinkedIn: <a href="https://www.linkedin.com/in/<your-profile>" style="color:var(--accent)">https://www.linkedin.com/in/&lt;your-profile&gt;</a><br />
        Email: <span class="muted">your.email@example.com</span></p>
      </div>

      <hr />

      <div>
        <h2>Suggested repo topics / tags</h2>
        <p class="muted">RAG, retrieval-augmented-generation, langchain, sentence-transformers, chromadb, embeddings, groq, semantic-search, nlp, python</p>
      </div>

      <footer>
        <div class="small"><strong>Short repo description:</strong> Production‑oriented Retrieval‑Augmented Generation pipeline with Sentence‑Transformers and ChromaDB (optional Groq backend).</div>
      </footer>
    </section>
  </div>
</body>
