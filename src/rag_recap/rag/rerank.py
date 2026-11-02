from rag_recap.config import settings, runner_settings

import requests as rq

def rerank_chunks(query, chunks):
    reranker_cfg = runner_settings['reranker']
    payload = {
        'model': reranker_cfg['model'],
        'query': query,
        'documents': [chunk.payload['text'] for chunk in chunks],
        'top_n': reranker_cfg['top_k']
    }

    r = rq.post(settings.rerank_url, json = payload)
    r.raise_for_status()
    return [(chunks[result['index']], result['relevance_score']) for result in r.json()['results']]