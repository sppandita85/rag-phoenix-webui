print("Starting PDF parsing...")
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat

import os
class PDFParser:
    def __init__(self):
        self.chunker = HybridChunker(max_tokens=2048)
        self.converter = DocumentConverter()
    
    def __getFileExtension(self,filename):
        file_extension = filename.split('.')[-1].lower()
        if file_extension == 'pdf':
            iformat = InputFormat.PDF
        elif file_extension == 'docx':
            iformat = InputFormat.DOCX
        else:
            print(f"Unsupported file format: {file_extension}. Skipping file.")
            iformat = None
        return iformat
    
    def __convertToMarkdown(self, doc):
        """
        Converts docling document object to Markdown format using native methods.
        
        :param doc: docling document object
        :return: Markdown formatted string
        """
        try:
            # Try to use docling's native markdown conversion first
            if hasattr(doc, 'export_to_markdown'):
                markdown_text = doc.export_to_markdown()
                print(f"Document converted to Markdown using native docling export_to_markdown method")
                return markdown_text
            elif hasattr(doc, 'document') and hasattr(doc.document, 'export_to_markdown'):
                markdown_text = doc.document.export_to_markdown()
                print(f"Document converted to Markdown using document.export_to_markdown()")
                return markdown_text
            
            # Fallback: try to extract text if markdown conversion is not available
            print(f"Native markdown conversion not available, extracting text content")
            if hasattr(doc, 'document') and hasattr(doc.document, 'text'):
                return doc.document.text
            elif hasattr(doc, 'text'):
                return doc.text
            else:
                return str(doc)
                
        except Exception as e:
            print(f"Error in markdown conversion: {e}")
            # Fallback to plain text extraction
            try:
                if hasattr(doc, 'document') and hasattr(doc.document, 'text'):
                    return doc.document.text
                elif hasattr(doc, 'text'):
                    return doc.text
                else:
                    return str(doc)
            except:
                return str(doc)
    
    def __traverseChunks(self, chunks, length_of_chunks=100):
        list_of_chunks = []
        for element in chunks:
            print(f"Chunk: {element.text[:length_of_chunks]}")
            list_of_chunks.append(element.text[:length_of_chunks])
        return list_of_chunks
    
    def parse(self,pdf_directory, length_of_chunks=100):
        """
        Parses the PDF file and returns the extracted text in Markdown format.
        
        :param pdf_directory: Path to the directory containing PDF/DOCX files.
        :param length_of_chunks: Maximum length of each text chunk.
        :return: List of text chunks in Markdown format.
        """
        print("-"*50)
        print("Parsing PDF/DOCX files to Markdown format...")
        print("-"*50)
        parsed_cv = []
        if not os.path.exists(pdf_directory):
            raise FileNotFoundError(f"The directory {pdf_directory} does not exist.")
        
        for filename in os.listdir(pdf_directory):
            file_path = os.path.join(pdf_directory,filename)
            iformat = self.__getFileExtension(filename)
            if iformat is None:
                continue
                
            print(f"Processing file: {file_path} with format {iformat}")
            doc = self.converter.convert(file_path)
            print(f"Document converted. Converting to Markdown format...")
            
            # Convert document to markdown format using native docling methods
            markdown_content = self.__convertToMarkdown(doc)
            print(f"Markdown conversion completed. Processing with chunker...")
            
            # Create a simple document object for chunking
            # The chunker expects a document-like object
            if hasattr(doc, 'document'):
                doc_for_chunking = doc.document
            else:
                doc_for_chunking = doc
            
            doc_chunks = self.chunker.chunk(dl_doc=doc_for_chunking)
            print(f"Chunking completed. Extracting text from chunks...")
            print(f"Printing Chunks for file: {filename}")
            
            list_of_chunks = self.__traverseChunks(doc_chunks, length_of_chunks=length_of_chunks)
            print(f"Total chunks extracted: {len(list_of_chunks)}")
            parsed_cv.append(list_of_chunks)
            
        print("Parsing completed.")
        print("-"*50)
        return parsed_cv