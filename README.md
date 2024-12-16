# mini-rag
RAG playground for locally running SLM.

How to use:

1. Clone the repo.
2. Install dependencies with `poetry install`.
3. Install _ollama_
4. Install relevant models in ollama (e.g. currently the code used _llama3.1:8b_ and _mxbai-embed-large_).
5. Create a folder `pdfs` inside the project and store some pdfs there
6. Run queries with `poetry run python3 ./main.py --query 'YOUR PROMPT' --reindex`
