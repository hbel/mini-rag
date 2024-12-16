from typing import Dict, List, Tuple
import ollama
from langchain_core.documents import Document

class OllamaModel:
    """Wrapper for our small language model"""
    
    def __init__(self, model_name: str = "llama3.1:8b"):
        self.model_name = model_name
        
    def generate_response(self, query: str, context: str) -> str:
        """
        Generate an answer based on the user's query and the provided context.
        
        Args:
            query: User query
            context: Context that can be used to generate a response (extracted from document metadata)
            
        Returns:
            Generated response from the model
        """
        prompt = f"""
        Based on the following Context, answer this query: {query}
        
        Context:
        {context}
        
        Answer:
        """
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False
            )
            
            answer = response.get('response', '')
            
            if not answer:
                print("Model hasn't answered")
                answer = "Sorry, I could not generate an answer"
                
            return answer
            
        except Exception as e:
            print(f"Error on answer generation: {str(e)}")
            raise
            
    def process_query(self, query: str, relevant_docs: List[Tuple[Document, float]]) -> Dict:
        """
        Process a user query and use the relevant documents as a context
        
        Args:
            query: User query
            relevant_docs: List of documents (with score) for the context
            
        Returns:
            Dict containing question, answer and used sources
        """
        context = "\n".join([doc.page_content for doc, _ in relevant_docs])
        
        answer = self.generate_response(query, context)
        
        return {
            "question": query,
            "answer": answer,
            "source_documents": [
                {"content": doc.page_content, "source": doc.metadata["source"], "score": score}
                for doc, score in relevant_docs
            ]
        }