from typing import List
from tqdm.auto import tqdm
import json
import ollama
from langchain_core.embeddings import Embeddings

class OllamaEmbeddings(Embeddings):
    """Adapter to provide Ollama embeddings to langchain"""
    
    # According to my research mxbai-embed-large currently provides the best embeddings.
    def __init__(self, model_name: str = "mxbai-embed-large"):
        self.model_name = model_name
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors from texts"""
        embeddings = []
        for text in tqdm(texts, desc="Generating embeddings", unit="chunk"):
            try:
                response = ollama.embeddings(model=self.model_name, prompt=text)
                
                if 'embedding' not in response:
                    print("Unexpected API result from ollama:")
                    print(json.dumps(response, indent=2))
                    raise ValueError("Unable to find embeddings ")
                    
                embeddings.append(response['embedding'])
                
            except Exception as e:
                print(f"Error on creating embeddings: {str(e)}")
                raise
                
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Create the embeddings for a user query (to make a similarity search in the vector db)"""
        return self.embed_documents([text])[0]