from typing import Dict
from pathlib import Path
from tqdm.auto import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from PyPDF2 import PdfReader

from embeddings import OllamaEmbeddings
from cache import DocumentCache
from model import OllamaModel

class MiniRAG:
    def __init__(self, pdf_directory: str, collection_name: str = "pdf_collection"):
        """
        Initialse RAG system
        
        Args:
            pdf_directory: Folder containing the pdfs to reference
            collection_name: Name of the collection in chromaDB
        """
        self.pdf_directory = Path(pdf_directory)
        self.collection_name = collection_name
        self.document_cache = DocumentCache()
        self.embedding_model = OllamaEmbeddings()
        self.llm = OllamaModel()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        
        try:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
                persist_directory="./chroma_db"
            )
            print(f"Loaded vector store with {self.vectorstore._collection.count()} documents")
        except Exception as e:
            print(f"Vector db access error: {e}")
            self.vectorstore = None

    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from given pdf."""
        # This is a very primitive way to get the text/metadata of the document!
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

    def index_documents(self) -> None:
        """Create an index of the documents in the pdf directory."""
        documents = []
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        new_or_modified_files = []
        
        print("Check for new and changed documents...")
        for pdf_file in pdf_files:
            if not self.document_cache.is_document_processed(pdf_file):
                new_or_modified_files.append(pdf_file)
            
        if not new_or_modified_files:
            print("No changes detected.")
            return
            
        print(f"{len(new_or_modified_files)} new or changed documents found.")
        for pdf_file in tqdm(new_or_modified_files, desc="Processing PDFs", unit="file"):
            text = self._extract_text_from_pdf(pdf_file)
            chunks = self.text_splitter.split_text(text)
            
            doc_chunks = [
                Document(
                    page_content=chunk,
                    metadata={"source": str(pdf_file.name)}
                ) for chunk in chunks
            ]
            documents.extend(doc_chunks)
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            collection_name=self.collection_name,
            persist_directory="./chroma_db"
        )
        
        for pdf_file in new_or_modified_files:
            self.document_cache.update_document(pdf_file)
        self.document_cache.save()

    def clear_vectorstore(self) -> None:
        """Clear vector store"""
        try:
            client = Chroma.PersistentClient(path="./chroma_db")
            
            try:
                client.delete_collection(name=self.collection_name)
                print(f"Existing collection '{self.collection_name}' erased.")
            except ValueError:
                print("No existing collection was found.")
                
            self.document_cache.clear()
            self.document_cache.save()
            
            self.vectorstore = None
            
            print("Vector store reset successfully")
            
        except Exception as e:
            print(f"Error resetting vector store: {e}")
            raise
    
    def delete_document(self, filename: str) -> None:
        """
        Deletes a specific document from the vector store.
        
        Args:
            filename: Name of PDF file
        """
        if self.vectorstore is not None:
            self.vectorstore._collection.delete(where={"source": filename})
            self.vectorstore.persist()
            
            full_path = str(self.pdf_directory / filename)
            self.document_cache.remove_document(full_path)
            self.document_cache.save()
            
            print(f"Document {filename} deleted from vector store.")
        else:
            print("No vector store found.")

    def list_documents(self) -> None:
        """Show all indexed documents."""
        if self.vectorstore is not None:
            results = self.vectorstore._collection.get()
            if results['metadatas']:
                sources = set(meta['source'] for meta in results['metadatas'] if 'source' in meta)
                print("\nIndexed documents:")
                for source in sorted(sources):
                    print(f"- {source}")
            else:
                print("No documents were indexed yet.")
        else:
            print("No vector store found.")

    def query(self, query: str, k: int = 5) -> Dict:
        """
        Queries the most relevant documents for a prompt. 
        
        Args:
            query: User query
            k: Number of documents to return
            
        Returns:
            Dict of search results
        """
        if self.vectorstore is None:
            raise ValueError("No vector store found.")
            
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            return self.llm.process_query(query, results)
            
        except Exception as e:
            print(f"Error in processing user query: {str(e)}")
            raise