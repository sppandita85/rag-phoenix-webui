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
        Converts docling document object to Markdown format.
        
        :param doc: docling document object
        :return: Markdown formatted string
        """
        try:
            # Convert document to Markdown
            markdown_text = doc.to_markdown()
            print(f"Document converted to Markdown format successfully")
            return markdown_text
        except AttributeError:
            # Fallback: try to extract text if to_markdown() is not available
            try:
                if hasattr(doc, 'document') and hasattr(doc.document, 'text'):
                    markdown_text = doc.document.text
                    print(f"Markdown conversion not available, using plain text extraction")
                    return markdown_text
                else:
                    # If no markdown conversion available, extract plain text
                    print("Markdown conversion not available, using plain text extraction")
                    if hasattr(doc, 'document') and hasattr(doc.document, 'text'):
                        return doc.document.text
                    elif hasattr(doc, 'text'):
                        return doc.text
                    else:
                        return str(doc)
            except Exception as e:
                print(f"Error in fallback text extraction: {e}")
                return str(doc)
        except Exception as e:
            print(f"Error converting to Markdown: {e}")
            # Fallback to plain text
            try:
                if hasattr(doc, 'document') and hasattr(doc.document, 'text'):
                    return doc.document.text
                elif hasattr(doc, 'text'):
                    return doc.text
                else:
                    return str(doc)
            except:
                return str(doc)
    
    def __text_to_markdown(self, text_content):
        """
        Converts plain text to basic markdown format.
        
        :param text_content: Plain text content
        :return: Markdown formatted string
        """
        try:
            # Basic markdown conversion
            markdown_text = text_content
            
            # Convert bullet points
            markdown_text = markdown_text.replace('●', '- ')
            
            # Convert line breaks to proper markdown
            markdown_text = markdown_text.replace('\n\n', '\n\n')
            
            # Add headers for sections (basic heuristic)
            lines = markdown_text.split('\n')
            markdown_lines = []
            
            for line in lines:
                line = line.strip()
                if line and len(line) < 100 and line.isupper() and not line.startswith('-'):
                    # Likely a header - make it markdown
                    markdown_lines.append(f"## {line}")
                else:
                    markdown_lines.append(line)
            
            return '\n'.join(markdown_lines)
            
        except Exception as e:
            print(f"Error in markdown conversion: {e}")
            return text_content
    
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
            
            # Extract text content from docling document
            if hasattr(doc, 'document') and hasattr(doc.document, 'text'):
                text_content = doc.document.text
            elif hasattr(doc, 'text'):
                text_content = doc.text
            else:
                text_content = str(doc)
            
            print(f"Text extraction completed. Converting to markdown...")
            
            # Convert text to markdown format
            markdown_content = self.__text_to_markdown(text_content)
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