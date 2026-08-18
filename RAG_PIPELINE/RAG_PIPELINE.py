"""
RAG Pipeline
============

An end-to-end Retrieval-Augmented Generation (RAG) pipeline built with
LangChain, Sentence-Transformers, and ChromaDB.

Workflow:
    1. Ingestion    - Load PDF documents from a folder.
    2. Chunking     - Split documents into overlapping text chunks.
    3. Embedding    - Convert chunks into dense vector embeddings.
    4. Storage      - Persist embeddings in a Chroma vector store.
    5. Retrieval    - Perform semantic search over the vector store
                       given a natural-language query.

To install the required dependencies:
    pip install langchain langchain-core langchain-community \
        langchain-text-splitters pypdf pymupdf sentence-transformers \
        chromadb scikit-learn
"""

# Cell 1: Imports
# All third-party and standard-library imports required by the pipeline.
import os
import uuid

import chromadb
from langchain_core.documents import Document
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
TEXT_DATA_PATH = "data/Python.txt"
PDF_FOLDER_PATH = "data/pdfs"
VECTOR_STORE_DIR = "data/vector_store"
VECTOR_STORE_COLLECTION = "pdf_documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# --------------------------------------------------------------------------
# Document Basics
# --------------------------------------------------------------------------
def demonstrate_document_basics():
    """
    Demonstrate the core LangChain `Document` object: how it is created
    and how its content/metadata are accessed.
    """
    # Cell 2: Create a sample Document to illustrate the schema used
    # throughout the pipeline (page_content + metadata).
    sample_doc = Document(
        page_content="Hello World!",
        metadata={"source": "https://www.google.com"},
    )

    # Cell 3: Inspect the sample document.
    # Expected output: Document(metadata={'source': 'https://www.google.com'},
    # page_content='Hello World!')
    print(sample_doc)

    # Cell 4: Confirm the object type.
    # Expected output: <class 'langchain_core.documents.base.Document'>
    print(type(sample_doc))

    return sample_doc


# --------------------------------------------------------------------------
# Document Loading (single-file example)
# --------------------------------------------------------------------------
def demonstrate_text_loading(path=TEXT_DATA_PATH):
    """
    Demonstrate loading a single plain-text file into LangChain
    `Document` objects using `TextLoader`.

    Note: alternative loaders for PDF ingestion (PyPDFLoader,
    PyMuPDFLoader) are available in langchain_community for single-file
    use cases; the pipeline below uses PyPDFLoader for batch PDF
    ingestion.
    """
    # Cell 5: Instantiate a text loader for a single .txt file.
    loader = TextLoader(path, encoding="utf-8")

    # Cell 6: Load the file into a list of Document objects.
    document = loader.load()

    # Cell 7: Inspect the loaded document(s).
    # Expected output: a list containing one Document with the file's
    # full text as page_content and {'source': path} as metadata.
    print(document)

    return document


# --------------------------------------------------------------------------
# Ingestion Pipeline: Documents
# --------------------------------------------------------------------------
def load_all_pdfs(folder_path=PDF_FOLDER_PATH):
    """
    Load every PDF file in `folder_path` into a single list of
    LangChain `Document` objects (one Document per page).
    """
    num_docs = 0
    all_docs = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            # Build the complete file path.
            pdf_path = os.path.join(folder_path, filename)

            loader = PyPDFLoader(pdf_path)
            doc = loader.load()

            all_docs.extend(doc)
            num_docs += 1

    print("total pdfs:", num_docs)
    print("total pages:", len(all_docs))
    return all_docs


# --------------------------------------------------------------------------
# Ingestion Pipeline: Chunking
# --------------------------------------------------------------------------
def split_docs(documents, chunk_size=500, chunk_overlap=50):
    """
    Split a list of Documents into smaller, overlapping chunks suitable
    for embedding, using a recursive character-based splitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunked_docs = text_splitter.split_documents(documents)
    return chunked_docs


# --------------------------------------------------------------------------
# Ingestion Pipeline: Embedding
# --------------------------------------------------------------------------
class EmbeddingManager:
    """
    Wraps a Sentence-Transformers model to generate dense vector
    embeddings for text chunks.
    """

    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        print("loading model....", self.model_name)
        self.model = SentenceTransformer(self.model_name)
        print("embedding dimensions=", self.model.get_sentence_embedding_dimension())

    def generate_embeddings(self, text):
        """Encode a string or list of strings into embedding vectors."""
        embeddings = self.model.encode(text, show_progress_bar=True)
        print("embeddings shape:", embeddings.shape)
        return embeddings


# --------------------------------------------------------------------------
# Ingestion Pipeline: Vector Store
# --------------------------------------------------------------------------
class VectorStoreManager:
    """
    Manages a persistent Chroma collection used to store document
    chunks alongside their embeddings and metadata.
    """

    def __init__(self, persist_directory=VECTOR_STORE_DIR, collection_name=VECTOR_STORE_COLLECTION):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.collection = None
        self.client = None

        self._initialize_store()

    def _initialize_store(self):
        """Create the persistent Chroma client and collection if needed."""
        os.makedirs(self.persist_directory, exist_ok=True)

        # Create a persistent client backed by local storage.
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # Create (or fetch) the collection used to store embeddings.
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "vector store collection for pdf embeddings in RAG"},
        )

        print("initialized the vector store with collection:", self.collection_name)
        print("docs in collection:", self.collection.count())

    def add_documents(self, documents, embeddings):
        """
        Add a batch of document chunks and their corresponding
        embeddings to the Chroma collection.
        """
        if len(documents) != len(embeddings):
            raise ValueError("num of documents does not match num of embeddings")

        # Assemble the parallel lists Chroma expects: ids, metadata,
        # document text, and embedding vectors.
        ids = []
        all_metadata = []
        documents_content = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4()}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            all_metadata.append(metadata)

            documents_content.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        # Write the full batch to the collection in a single call.
        self.collection.add(
            ids=ids,
            metadatas=all_metadata,
            documents=documents_content,
            embeddings=embeddings_list,
        )

        print("total documents added in vector store=", len(documents_content))
        print("docs in collection:", self.collection.count())


# --------------------------------------------------------------------------
# Retrieval Pipeline
# --------------------------------------------------------------------------
class RAGRetriever:
    """
    Performs semantic search over a vector store: embeds a natural
    language query and retrieves the most similar document chunks.
    """

    def __init__(self, embedding_manager, vector_store):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store

    def retrieve(self, query, top_k=5, score_threshold=0.0):
        """
        Retrieve the top-k document chunks most similar to `query`,
        filtered by a minimum cosine similarity score.
        """
        # Embed the query using the same model used for the documents.
        query_embeddings = self.embedding_manager.generate_embeddings([query])[0]

        # Run a semantic (nearest-neighbor) search against the collection.
        results = self.vector_store.collection.query(
            query_embeddings=[query_embeddings.tolist()],
            n_results=top_k,
        )

        # Convert Chroma's distance-based results into ranked, scored hits.
        retrieved_docs = []

        if results["documents"] and results["documents"][0]:
            ids = results["ids"][0]
            metadatas = results["metadatas"][0]
            documents = results["documents"][0]
            distances = results["distances"][0]

            for i, (doc_id, metadata, document, distance) in enumerate(
                zip(ids, metadatas, documents, distances)
            ):
                similarity_score = 1 - distance

                if similarity_score >= score_threshold:
                    retrieved_docs.append(
                        {
                            "id": doc_id,
                            "document": document,
                            "metadata": metadata,
                            "distance": distance,
                            "similarity_score": similarity_score,
                            "rank": i + 1,
                        }
                    )

            print(f"retrieved {len(retrieved_docs)} documents")
        else:
            print("no documents found")

        return retrieved_docs


# --------------------------------------------------------------------------
# Main: build and query the RAG pipeline end-to-end
# --------------------------------------------------------------------------
def main():
    # Introductory examples of the core LangChain Document abstraction.
    demonstrate_document_basics()
    demonstrate_text_loading()

    # Ingestion: Data -> Documents -> Chunks.
    all_pdf_documents = load_all_pdfs()
    # Expected output: type(all_pdf_documents[1]) -> <class
    # 'langchain_core.documents.base.Document'>

    chunks = split_docs(all_pdf_documents)
    print("total chunks:", len(chunks))

    # Ingestion: Chunks -> Embeddings -> Vector Store.
    embedding_manager = EmbeddingManager()
    vector_store = VectorStoreManager()

    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)
    vector_store.add_documents(chunks, embeddings)

    # Retrieval: Query -> Embedding -> Semantic Search.
    rag_retriever = RAGRetriever(embedding_manager, vector_store)
    results = rag_retriever.retrieve("What is encoder decoder")

    for result in results:
        print(result)


if __name__ == "__main__":
    main()