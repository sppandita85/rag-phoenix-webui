print("Starting this class ...")
import os
import torch
import gc
from llama_index.core import (
    SimpleDirectoryReader, 
    VectorStoreIndex, 
    StorageContext,
    Document
)
from llama_index.core.settings import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.readers.file import PyMuPDFReader
from llama_index.llms.ollama import Ollama
from llama_index.core.indices.vector_store import VectorStoreIndex

# Import docling for PDF parsing and chunking
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(os.path.join(project_root, 'pdf_parsing'))
from pdf_parser import PDFParser

# database Connection
# from postgres.postgress_connection import create_connection

# from huggingface_hub import login
# login(token="YOUR TOKEN")
print("Starting this class  1 ...")

DATABASE_NAME = 'poc'
DATABASE_USER = 'sppandita85'
DATABASE_PASSWORD = 'password123'
DATABASE_HOST = 'localhost'
DATABASE_PORT = 5432
EMBED_DIM = 1024

class LlamaModel:
    def __init__(self, embedding_model='qllama/bge-large-en-v1.5', llm_model='llama3.2', data_directory='inputs/input_data'):
        print("Initializing LlamaModel with Ollama...")
        
        # Clear GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        try:
            # Initialize embedding model using Ollama
            Settings.embed_model = OllamaEmbedding(
                model_name=embedding_model,
                base_url="http://localhost:11434"
            )
            print(f"✅ Ollama embedding model '{embedding_model}' loaded successfully")
        except Exception as e:
            print(f"❌ Error loading Ollama embedding model: {e}")
            raise
        
        try:
            # Initialize the LLM using Ollama
            print(f"Loading Ollama LLM model: {llm_model}")
            
            llm = Ollama(
                model=llm_model,
                base_url="http://localhost:11434",
                request_timeout=120.0,
                temperature=0.7
            )
            Settings.llm = llm
            print(f"✅ Ollama LLM model '{llm_model}' loaded successfully")
        except Exception as e:
            print(f"❌ Error loading Ollama LLM model: {e}")
            raise
        
        print("LlamaModel initialized with Ollama embedding and LLM settings.")
        
        try:
            # Initialize vector store
            self.vector_store = PGVectorStore.from_params(
                database= DATABASE_NAME,
                host=DATABASE_HOST,
                port=DATABASE_PORT,
                user=DATABASE_USER,
                password=DATABASE_PASSWORD,
                embed_dim=EMBED_DIM,
                table_name='cv_embeddings'
            )
            print("✅ Vector store initialized successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize vector store: {e}")
            print("   Continuing with in-memory storage...")
            self.vector_store = None
        
        # Fix the data directory path to be relative to the project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.data_directory = os.path.join(project_root, data_directory)
        
        print(f"Loading documents from: {self.data_directory}")
        
        if not os.path.exists(self.data_directory):
            raise ValueError(f"Data directory does not exist: {self.data_directory}")
        
        try:
            # Use docling for PDF parsing and chunking instead of SimpleDirectoryReader
            self.documents = self._load_documents_with_docling()
            print(f"✅ Loaded {len(self.documents)} documents successfully using docling")
        except Exception as e:
            print(f"❌ Error loading documents: {e}")
            raise
        
        # Initialize storage context
        if self.vector_store:
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        else:
            self.storage_context = StorageContext.from_defaults()
        
        self.index = None
        self.query_engine = None
    
    def _load_documents_with_docling(self):
        """
        Load documents using docling for PDF parsing and chunking.
        Returns a list of LlamaIndex Document objects.
        """
        print("🔄 Using docling for PDF parsing and chunking...")
        
        # Initialize docling PDF parser
        pdf_parser = PDFParser()
        
        # Parse PDFs using docling (this will use your optimized max_tokens=2048)
        parsed_chunks = pdf_parser.parse(self.data_directory, length_of_chunks=100)
        
        # Convert docling chunks to LlamaIndex Document objects
        llama_documents = []
        
        for file_chunks in parsed_chunks:
            for i, chunk_text in enumerate(file_chunks):
                # Create LlamaIndex Document object
                doc = Document(
                    text=chunk_text,
                    metadata={
                        "source": "docling_parser",
                        "chunk_id": i,
                        "chunk_size": len(chunk_text),
                        "parser": "docling_hybrid_chunker",
                        "chunk_type": "markdown_chunk",
                        "max_tokens": 2048,
                        "processing_pipeline": "docling_pdf_parser -> markdown_conversion -> hybrid_chunking -> llama_index"
                    }
                )
                llama_documents.append(doc)
        
        print(f"📊 Docling processing complete:")
        print(f"   - Files processed: {len(parsed_chunks)}")
        print(f"   - Total chunks created: {len(llama_documents)}")
        print(f"   - Chunk size setting: max_tokens=2048")
        print(f"   - Content format: Markdown")
        print(f"   - Processing: PDF → Text → Markdown → Chunks → Embeddings")
        
        return llama_documents

    def create_index(self):
        print("Creating index...")
        try:
            self.index = VectorStoreIndex.from_documents(
                self.documents, 
                storage_context=self.storage_context,
                show_progress=True
            )
            print("Index created successfully.")
            print(f"Total documents indexed: {len(self.documents)}")
            print(f"Index summary: {self.index.summary}")    
            print("-" * 50)
            
            if Settings.llm:
                self.query_engine = self.index.as_query_engine()
                print("✅ Query engine initialized successfully")
            else:
                print("⚠️  Query engine not available (no LLM)")
                
        except Exception as e:
            print(f"❌ Error creating index: {e}")
            raise

    def query(self,query_text):
        if not self.query_engine:
            return "Query engine not available. Please check LLM initialization."
        
        try:
            response = self.query_engine.query(query_text)
            return response
        except Exception as e:
            return f"Error during query: {e}"

#llama_model = LlamaModel()  # Create an instance
#response = LlamaModel.query(query_text="Who among the two candidates is more suitable for a typist job")
#print("the execution has reached here")
#print(response)
    