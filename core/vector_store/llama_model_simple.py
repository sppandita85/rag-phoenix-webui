#!/usr/bin/env python3
"""
Simplified LlamaModel with memory-efficient settings.
"""

import os
import time
from llama_index.core import (
    SimpleDirectoryReader, 
    VectorStoreIndex, 
    StorageContext, 
)
from llama_index.core.settings import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.vector_stores.postgres import PGVectorStore
from transformers import AutoTokenizer, AutoModelForCausalLM

class LlamaModelSimple:
    def __init__(self, model_name='qllama/bge-large-en-v1.5:latest', data_directory='inputs/input_data'):
        print("Initializing LlamaModelSimple with Ollama embeddings...")
        
        # Use Ollama embedding model
        Settings.embed_model = OllamaEmbedding(
            model_name=model_name,
            base_url="http://localhost:11434"
        )
        
        # Use a smaller, more memory-efficient LLM
        model_id = "microsoft/DialoGPT-small"  # Much smaller than phi-2
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            llm = HuggingFaceLLM(
                tokenizer=tokenizer,
                model_name=model_id,
                device_map="cpu",
                max_new_tokens=512,  # Limit output length
                context_window=1024,  # Reduce context window
                generate_kwargs={"do_sample": True, "temperature": 0.7}
            )
            Settings.llm = llm
            print("✅ LlamaModelSimple initialized with Ollama embeddings.")
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize LLM: {e}")
            print("   Continuing without LLM for now...")
            Settings.llm = None
        
        # Fix the data directory path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.data_directory = os.path.join(project_root, data_directory)
        
        print(f"Loading documents from: {self.data_directory}")
        
        if not os.path.exists(self.data_directory):
            raise ValueError(f"Data directory does not exist: {self.data_directory}")
        
        # Load documents
        self.documents = SimpleDirectoryReader(
            self.data_directory,
        ).load_data()
        
        print(f"✅ Loaded {len(self.documents)} documents successfully")
        
        # Initialize PostgreSQL storage and create index
        print("🔄 Setting up PostgreSQL vector store...")
        self.storage_context = self._setup_postgres_storage()
        
        # Create index immediately after loading documents
        print("🔄 Creating vector index and storing embeddings...")
        self.create_index()
    
    def _ensure_table_exists(self):
        """Ensure the data_cv_embeddings table exists with correct structure."""
        try:
            import psycopg2
            
            # Connect to database
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="poc",
                user="sppandita85",
                password=""
            )
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'data_cv_embeddings'
                );
            """)
            
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                print("🔄 Creating data_cv_embeddings table...")
                cursor.execute("""
                    CREATE TABLE data_cv_embeddings (
                        id BIGSERIAL PRIMARY KEY,
                        text TEXT NOT NULL,
                        metadata_ JSONB,
                        node_id VARCHAR,
                        embedding vector(1024)
                    );
                """)
                
                # Create index
                cursor.execute("""
                    CREATE INDEX cv_embeddings_idx_1 
                    ON data_cv_embeddings 
                    USING GIN ((metadata_ ->> 'ref_doc_id'));
                """)
                
                conn.commit()
                print("✅ data_cv_embeddings table created successfully")
            else:
                print("✅ data_cv_embeddings table already exists")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"⚠️  Warning: Could not ensure table exists: {e}")
    
    def _setup_postgres_storage(self):
        """Set up PostgreSQL vector store storage context."""
        try:
            print(f"🔄 Configuring PostgreSQL vector store for table: data_cv_embeddings")
            
            # Ensure the target table exists
            self._ensure_table_exists()
            
            # Configure PostgreSQL vector store with explicit table name
            vector_store = PGVectorStore.from_params(
                host="localhost",
                port=5432,
                database="poc",
                user="sppandita85",
                password="",  # No password for local development
                table_name="data_cv_embeddings",  # Force use of this specific table
                embed_dim=1024,  # Dimension for qllama/bge-large-en-v1.5
            )
            
            # Verify the table name was set correctly
            print(f"✅ Vector store configured with table: {vector_store.table_name}")
            
            # Create storage context with PostgreSQL
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            print("✅ PostgreSQL vector store configured successfully")
            return storage_context
            
        except Exception as e:
            print(f"⚠️  Warning: Could not configure PostgreSQL: {e}")
            print("   Falling back to in-memory storage...")
            return StorageContext.from_defaults()
    
    def _copy_data_to_target_table(self):
        """Copy data from auto-created table to data_cv_embeddings."""
        try:
            import psycopg2
            
            # Connect to database
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="poc",
                user="sppandita85",
                password=""
            )
            cursor = conn.cursor()
            
            # Find which table has the data
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE '%cv_embeddings%'
                AND table_name != 'data_cv_embeddings'
                AND table_name != 'data_cv_embeddings_old'
                ORDER BY table_name;
            """)
            
            source_tables = cursor.fetchall()
            
            for (source_table,) in source_tables:
                # Check if source table has data
                cursor.execute(f"SELECT COUNT(*) FROM {source_table}")
                count = cursor.fetchone()[0]
                
                if count > 0:
                    print(f"🔄 Found {count} records in {source_table}, copying to data_cv_embeddings...")
                    
                    # Copy data to target table
                    cursor.execute(f"""
                        INSERT INTO data_cv_embeddings (text, metadata_, node_id, embedding)
                        SELECT text, metadata_, node_id, embedding FROM {source_table}
                        ON CONFLICT DO NOTHING;
                    """)
                    
                    # Get the count of copied records
                    cursor.execute("SELECT COUNT(*) FROM data_cv_embeddings")
                    final_count = cursor.fetchone()[0]
                    
                    print(f"✅ Copied {count} records to data_cv_embeddings (total: {final_count})")
                    
                    # Drop the source table
                    cursor.execute(f"DROP TABLE IF EXISTS {source_table} CASCADE")
                    print(f"🗑️  Dropped source table: {source_table}")
                    
                    break
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"⚠️  Warning: Could not copy data to target table: {e}")
    
    def create_index(self):
        print("Creating index with Ollama embeddings...")
        try:
            # Debug: Show which table we're using
            if hasattr(self.storage_context, 'vector_store') and hasattr(self.storage_context.vector_store, 'table_name'):
                print(f"🔍 Using table: {self.storage_context.vector_store.table_name}")
                
                # Force the table name if it's wrong
                if self.storage_context.vector_store.table_name != "data_cv_embeddings":
                    print(f"⚠️  Table name mismatch! Forcing to data_cv_embeddings")
                    self.storage_context.vector_store.table_name = "data_cv_embeddings"
                    print(f"✅ Table name forced to: {self.storage_context.vector_store.table_name}")
            else:
                print("⚠️  Could not determine table name from storage context")
            
            self.index = VectorStoreIndex.from_documents(
                self.documents, 
                storage_context=self.storage_context,
                show_progress=True
            )
            print("✅ Index created successfully.")
            print(f"Total documents indexed: {len(self.documents)}")
            
            # Copy data to the target table
            self._copy_data_to_target_table()
            
            if Settings.llm:
                self.query_engine = self.index.as_query_engine()
                print("✅ Query engine initialized.")
            else:
                print("⚠️  Query engine not available (no LLM)")
                
        except Exception as e:
            print(f"❌ Error creating index: {e}")
            raise
    
    def query(self, query_text):
        if not self.query_engine:
            return "Query engine not available. Please check LLM initialization."
        
        try:
            response = self.query_engine.query(query_text)
            return response
        except Exception as e:
            return f"Error during query: {e}"
    
    def get_document_summary(self):
        """Get a summary of loaded documents without requiring the full index."""
        summary = []
        for i, doc in enumerate(self.documents):
            doc_info = {
                'index': i,
                'file_path': doc.metadata.get('file_path', 'Unknown'),
                'text_length': len(doc.text),
                'preview': doc.text[:200] + "..." if len(doc.text) > 200 else doc.text
            }
            summary.append(doc_info)
        return summary

