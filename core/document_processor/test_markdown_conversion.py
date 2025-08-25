#!/usr/bin/env python3
"""
Test script for the updated PDFParser with Markdown conversion.
"""

from pdf_parser import PDFParser
import os

def test_markdown_conversion():
    """Test the Markdown conversion functionality."""
    
    # Initialize the parser
    parser = PDFParser()
    
    # Test with a sample directory (you can change this path)
    test_directory = "../inputs/CV"  # Assuming you have some PDF files in inputs/CV
    
    print("Testing PDFParser with Markdown conversion...")
    print(f"Looking for files in: {test_directory}")
    
    if os.path.exists(test_directory):
        try:
            # Parse documents with Markdown conversion
            results = parser.parse(test_directory, length_of_chunks=150)
            
            print(f"\n✅ Parsing completed successfully!")
            print(f"📄 Processed {len(results)} document(s)")
            
            # Display results
            for i, doc_chunks in enumerate(results):
                print(f"\n📋 Document {i+1}:")
                print(f"   Number of chunks: {len(doc_chunks)}")
                for j, chunk in enumerate(doc_chunks[:3]):  # Show first 3 chunks
                    print(f"   Chunk {j+1}: {chunk[:100]}...")
                if len(doc_chunks) > 3:
                    print(f"   ... and {len(doc_chunks) - 3} more chunks")
                    
        except Exception as e:
            print(f"❌ Error during parsing: {e}")
    else:
        print(f"❌ Test directory not found: {test_directory}")
        print("Please ensure you have some PDF/DOCX files in the inputs/CV directory")

if __name__ == "__main__":
    test_markdown_conversion()

