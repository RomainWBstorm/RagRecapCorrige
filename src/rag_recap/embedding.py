from rag_recap.config import settings

from openai import OpenAI
from qdrant_client import models


def get_embedder():
    return OpenAI(
        base_url = settings.embedding_url,
        api_key = settings.llm_server_api_key
    )

def get_sparse_options(index_cfg):
    return {
        'language': settings.corpus_language,
        'ascii_folding': index_cfg['sparse_embedder_ascii_folding'],
    }

def get_query_text(query, embedder_cfg):
    if embedder_cfg['query_instruction']:
        return f"Instruct: {embedder_cfg['query_instruction']}\nQuery: {query}"
    return query

def embed(chunks, embedder, model_name):
    response = embedder.embeddings.create(model=model_name, input=chunks)
    return [item.embedding for item in response.data]

def embed_sparse(text, options):
    return models.Document(text=text, model='qdrant/bm25', options=options)
