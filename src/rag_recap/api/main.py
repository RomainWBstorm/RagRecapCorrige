from rag_recap.qdrant import get_qdrant_client
from rag_recap.embedding import get_embedder
from rag_recap.rag.ask import generate_response, get_llm_client

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title='RAG Recap')

qdrant_client = get_qdrant_client()
embedder = get_embedder()
llm_client = get_llm_client()

class Question(BaseModel):
    question: str

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/ask')
async def ask(question: Question):
    answer, llm_metrics, chunks = await generate_response(question.question, qdrant_client, embedder, llm_client)
    return {
        'answer': answer,
        'sources': [
            {'document': chunk.payload['document'], 'chunk_index': chunk.payload['chunk_index'], 'score': score}
            for chunk, score in chunks
        ],
    }
