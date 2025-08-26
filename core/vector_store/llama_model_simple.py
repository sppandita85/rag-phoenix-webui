#!/usr/bin/env python3
"""
Simplified LlamaModel with docling integration and memory-efficient settings.
"""

import os
import time
from llama_index.core import (
    VectorStoreIndex, 
    StorageContext, 
    Document
)
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.postgres import PGVectorStore

class LlamaModelSimple:
    def __init__(self, data_directory='inputs/input_data'):
        print("Initializing LlamaModelSimple with docling integration...")
        
        # Use qllama/bge-large-en-v1.5 for embeddings from Ollama
        try:
            from llama_index.embeddings.ollama import OllamaEmbedding
            Settings.embed_model = OllamaEmbedding(
                model_name="qllama/bge-large-en-v1.5",
                base_url="http://localhost:11434"
            )
            print("✅ Ollama embedding model (qllama/bge-large-en-v1.5) initialized")
        except Exception as e:
            print(f"❌ Error initializing embedding model: {e}")
            raise
        
        # Use llama3.2 for LLM
        try:
            llm = Ollama(
                model="llama3.2",
                base_url="http://localhost:11434",
                request_timeout=120.0
            )
            Settings.llm = llm
            print("✅ Ollama LLM (llama3.2) initialized")
        except Exception as e:
            print(f"❌ Error initializing LLM: {e}")
            raise
        
        # Fix the data directory path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.data_directory = os.path.join(project_root, data_directory)
        
        print(f"Loading documents from: {self.data_directory}")
        
        if not os.path.exists(self.data_directory):
            raise ValueError(f"Data directory does not exist: {self.data_directory}")
        
        # Process documents using docling
        print("🔄 Processing documents with docling...")
        self.documents = self._process_documents_with_docling()
        
        print(f"✅ Processed {len(self.documents)} documents successfully")
        
        # Initialize PostgreSQL storage and create index
        print("🔄 Setting up PostgreSQL vector store...")
        self.storage_context = self._setup_postgres_storage()
        
        # Create index immediately after loading documents
        print("🔄 Creating vector index and storing embeddings...")
        self.create_index()
    
    def _process_documents_with_docling(self):
        """Process PDF documents using docling library."""
        try:
            from core.document_processor.pdf_parser import PDFParser
            
            # Initialize docling parser
            parser = PDFParser()
            
            # Parse documents using docling
            parsed_chunks = parser.parse(self.data_directory, length_of_chunks=1000)
            
            # Convert chunks to llama_index Document objects
            documents = []
            for file_chunks in parsed_chunks:
                for i, chunk_text in enumerate(file_chunks):
                    # Create a Document object for each chunk
                    doc = Document(
                        text=chunk_text,
                        metadata={
                            'chunk_id': i,
                            'source': 'docling_processed',
                            'chunk_type': 'text'
                        }
                    )
                    documents.append(doc)
            
            print(f"✅ Created {len(documents)} document chunks from docling processing")
            return documents
            
        except Exception as e:
            print(f"❌ Error processing documents with docling: {e}")
            raise
    
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
                    CREATE INDEX data_cv_embeddings_idx_1 
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
            print(f"❌ Error ensuring table exists: {e}")
            raise
    
    def _setup_postgres_storage(self):
        """Set up PostgreSQL vector store storage context."""
        try:
            print(f"🔄 Configuring PostgreSQL vector store for table: data_cv_embeddings")
            
            # Ensure the target table exists
            self._ensure_table_exists()
            
            # Truncate the table before storing new embeddings
            import psycopg2
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="poc",
                user="sppandita85",
                password=""
            )
            cursor = conn.cursor()
            print("🔄 Truncating data_cv_embeddings table before storing new embeddings...")
            cursor.execute("TRUNCATE data_cv_embeddings RESTART IDENTITY CASCADE;")
            conn.commit()
            cursor.close()
            conn.close()
            print("✅ Table truncated successfully")
            
            # Configure PostgreSQL vector store with explicit table name
            vector_store = PGVectorStore.from_params(
                host="localhost",
                port=5432,
                database="poc",
                user="sppandita85",
                password="",  # No password for local development
                table_name="data_cv_embeddings",  # Use data_cv_embeddings table
                embed_dim=1024,  # Dimension for qllama/bge-large-en-v1.5
            )
            
            # Verify the table name was set correctly
            print(f"✅ Vector store configured with table: {vector_store.table_name}")
            
            # Create storage context with PostgreSQL
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Store vector store reference for later use
            self.vector_store = vector_store
            
            print("✅ PostgreSQL vector store configured successfully")
            return storage_context
            
        except Exception as e:
            print(f"❌ Error configuring PostgreSQL: {e}")
            raise
    
    def create_index(self):
        print("Creating index with qllama/bge-large-en-v1.5 embeddings...")
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
            
            print("🔄 Starting document indexing and embedding generation...")
            print(f"📄 Processing {len(self.documents)} documents with qllama/bge-large-en-v1.5...")
            
            self.index = VectorStoreIndex.from_documents(
                self.documents, 
                storage_context=self.storage_context,
                show_progress=True
            )
            
            # Manually add documents to ensure persistence
            print("🔄 Manually persisting documents to vector store...")
            for i, doc in enumerate(self.documents):
                try:
                    # Handle docling document format (tuples)
                    if isinstance(doc, tuple):
                        # Extract text from tuple format
                        doc_text = doc[0] if len(doc) > 0 else str(doc)
                        doc_metadata = {"source": "docling", "chunk_id": i}
                    else:
                        # Handle LlamaIndex Document objects
                        doc_text = doc.text if hasattr(doc, 'text') else str(doc)
                        doc_metadata = doc.metadata if hasattr(doc, 'metadata') else {"source": "docling", "chunk_id": i}
                    
                    # Create a proper TextNode for persistence (vector store expects BaseNode)
                    from llama_index.core.schema import TextNode
                    persist_node = TextNode(
                        text=doc_text,
                        metadata=doc_metadata
                    )
                    
                    # Generate embedding for the node before adding to vector store
                    print(f"🔄 Generating embedding for document {i+1}/{len(self.documents)}...")
                    try:
                        # Use the embedding model to generate embeddings
                        embedding = Settings.embed_model.get_text_embedding(doc_text)
                        persist_node.embedding = embedding
                        print(f"✅ Embedding generated for document {i+1}")
                    except Exception as embed_error:
                        print(f"❌ Failed to generate embedding: {embed_error}")
                        continue
                    
                    # Add to vector store using the correct method
                    if hasattr(self.vector_store, 'add'):
                        # The add method expects List[BaseNode]
                        try:
                            self.vector_store.add([persist_node])
                            print(f"✅ Persisted document {i+1}/{len(self.documents)} using add()")
                            
                            # Force commit to database
                            if hasattr(self.vector_store, '_conn') and self.vector_store._conn:
                                self.vector_store._conn.commit()
                                print(f"✅ Committed document {i+1} to database")
                            
                        except Exception as add_error:
                            print(f"❌ add() method failed: {add_error}")
                            break
                    else:
                        print(f"⚠️  Vector store missing 'add' method")
                        break
                        
                except Exception as e:
                    print(f"❌ Error persisting document {i+1}: {e}")
                    print(f"🔍 Document type: {type(doc)}")
                    print(f"🔍 Document content: {str(doc)[:100]}...")
                    break
            
            # Force direct database persistence as backup
            print("🔄 Forcing direct database persistence...")
            try:
                import psycopg2
                conn = psycopg2.connect(
                    host="localhost",
                    port=5432,
                    database="poc",
                    user="sppandita85",
                    password=""
                )
                cursor = conn.cursor()
                
                # Insert embeddings directly into database
                for i, doc in enumerate(self.documents):
                    try:
                        # Extract text
                        if isinstance(doc, tuple):
                            doc_text = doc[0] if len(doc) > 0 else str(doc)
                        else:
                            doc_text = doc.text if hasattr(doc, 'text') else str(doc)
                        
                        # Generate embedding
                        embedding = Settings.embed_model.get_text_embedding(doc_text)
                        
                        # Insert into database
                        cursor.execute("""
                            INSERT INTO data_cv_embeddings (text, metadata_, embedding) 
                            VALUES (%s, %s, %s)
                        """, (doc_text, '{"source": "docling", "chunk_id": ' + str(i) + '}', embedding))
                        
                        print(f"✅ Direct DB insert: document {i+1}/{len(self.documents)}")
                        
                    except Exception as e:
                        print(f"❌ Direct DB insert failed for document {i+1}: {e}")
                        continue
                
                # Commit all changes
                conn.commit()
                print("✅ All embeddings committed to database")
                
                # Verify storage
                cursor.execute("SELECT COUNT(*) FROM data_cv_embeddings")
                count = cursor.fetchone()[0]
                print(f"✅ Database now contains {count} embeddings")
                
                cursor.close()
                conn.close()
                
            except Exception as e:
                print(f"❌ Direct database persistence failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Verify embeddings were stored
            import psycopg2
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="poc",
                user="sppandita85",
                password=""
            )
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM data_cv_embeddings")
            count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            if count > 0:
                print(f"✅ Index created successfully with {count} embeddings stored in data_cv_embeddings table.")
            else:
                print(f"⚠️  Warning: {count} embeddings found in database. Checking for errors...")
                
                # Check table structure
                cursor.execute("""
                    SELECT column_name, data_type, is_nullable 
                    FROM information_schema.columns 
                    WHERE table_name = 'data_cv_embeddings'
                    ORDER BY ordinal_position
                """)
                columns = cursor.fetchall()
                print("🔍 Table structure:")
                for col in columns:
                    print(f"   {col[0]}: {col[1]} (nullable: {col[2]})")
                
                # Check for any recent errors in PostgreSQL logs
                cursor.execute("""
                    SELECT * FROM data_cv_embeddings 
                    LIMIT 1
                """)
                sample = cursor.fetchone()
                print(f"🔍 Sample row: {sample}")
                
                # Try to insert a test embedding manually
                print("🔄 Testing manual insertion...")
                try:
                    test_vector = [0.1] * 1024  # 1024-dimensional test vector
                    cursor.execute("""
                        INSERT INTO data_cv_embeddings (text, metadata_, embedding) 
                        VALUES (%s, %s, %s)
                    """, ("test", "{}", test_vector))
                    conn.commit()
                    print("✅ Manual insertion successful - table structure is correct")
                    
                    # Clean up test data
                    cursor.execute("DELETE FROM data_cv_embeddings WHERE text = 'test'")
                    conn.commit()
                    
                except Exception as e:
                    print(f"❌ Manual insertion failed: {e}")
                    print("🔍 This suggests a table structure or pgvector issue")
            
            print(f"Total documents indexed: {len(self.documents)}")
            
            if Settings.llm:
                self.query_engine = self.index.as_query_engine()
                print("✅ Query engine initialized.")
            else:
                print("⚠️  Query engine not available (no LLM)")
                
        except Exception as e:
            print(f"❌ Error creating index: {e}")
            import traceback
            traceback.print_exc()
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
                'text_length': len(doc.text),
                'preview': doc.text[:200] + "..." if len(doc.text) > 200 else doc.text
            }
            summary.append(doc_info)
        return summary