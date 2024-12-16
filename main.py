import argparse
from rag import MiniRAG

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='RAG System für PDF Dokumente')
    parser.add_argument('--dir', type=str, default='./pdfs',
                      help='Verzeichnis mit den PDF-Dateien (default: ./pdfs)')
    parser.add_argument('--query', type=str,
                      help='Suchanfrage an das System')
    parser.add_argument('--reindex', action='store_true',
                      help='Erzwingt eine Neuindizierung aller Dokumente')
    parser.add_argument('--delete', type=str,
                      help='Löscht ein bestimmtes Dokument aus dem Vectorstore (Dateiname angeben)')
    parser.add_argument('--clear', action='store_true',
                      help='Löscht alle Dokumente aus dem Vectorstore')
    parser.add_argument('--list', action='store_true',
                      help='Zeigt alle indizierten Dokumente an')
    
    args = parser.parse_args()
    
    # System initialisieren
    rag = MiniRAG(pdf_directory=args.dir)

    # Perform actions as requested
    if args.delete:
        rag.delete_document(args.delete)
        return
        
    if args.list:
        rag.list_documents()
        return
    
    if args.clear:
        rag.clear_vectorstore()
        return

    if args.reindex or rag.vectorstore is None:
        print("Building document index...")
        rag.index_documents()
    
    # If a query was passed
    if args.query:
        try:
            result = rag.query(args.query)
            
            print("\nAnswer:", result["answer"])
            print("\nSources:")
            for doc in result["source_documents"]:
                print(f"- {doc['source']} (Score: {doc['score']:.2f})")
        except ValueError as e:
            print(f"Errors: {e}")
    else:
        print("No query provided. Please provide a query using the --query flag.")

if __name__ == "__main__":
    main()