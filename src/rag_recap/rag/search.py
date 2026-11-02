from rag_recap.config import runner_settings
from rag_recap.embedding import get_embedder, get_sparse_options, get_query_text, embed, embed_sparse
from rag_recap.qdrant import get_qdrant_client, get_collection_name

from qdrant_client import models


def dense_search(dense_vector, collection_name, qdrant_client, limit, query_filter = None):
    return qdrant_client.query_points(
        collection_name, query = dense_vector, using = 'dense',
        limit = limit, query_filter = query_filter
    ).points

def sparse_search(sparse_query, collection_name, qdrant_client, limit, query_filter = None):
    return qdrant_client.query_points(
        collection_name, query = sparse_query, using = 'sparse',
        limit = limit, query_filter = query_filter
    ).points

def hybrid_search(dense_vector, sparse_query, collection_name, qdrant_client, limit, query_filter = None):
    return qdrant_client.query_points(
        collection_name,
        prefetch = [
            models.Prefetch(query = dense_vector, using = 'dense', limit = limit),
            models.Prefetch(query = sparse_query, using = 'sparse', limit = limit),
        ],
        query = models.RrfQuery(rrf = models.Rrf(k = 60)),
        limit = limit, query_filter = query_filter
    ).points

def search_chunks(query, qdrant_client, embedder, query_filter = None):
    embedder_cfg = runner_settings['embedder']
    chunking_cfg = runner_settings['chunking']
    index_cfg = runner_settings['index']
    search_cfg = runner_settings['search']

    collection_name = get_collection_name(
        embedder_name = embedder_cfg['model'],
        chunk_type = chunking_cfg['type'],
        chunk_size = chunking_cfg['max_tokens'],
        sparse_vector = index_cfg.get('sparse_vector'),
        patient_line = chunking_cfg['patient_line'],
    )
    limit = search_cfg['candidates']

    if search_cfg['method'] == 'dense':
        dense_vector = embed([get_query_text(query, embedder_cfg)], embedder, embedder_cfg['model'])[0]
        return dense_search(dense_vector, collection_name, qdrant_client, limit, query_filter)

    if search_cfg['method'] == 'sparse':
        sparse_query = embed_sparse(query, get_sparse_options(index_cfg))
        return sparse_search(sparse_query, collection_name, qdrant_client, limit, query_filter)

    dense_vector = embed([get_query_text(query, embedder_cfg)], embedder, embedder_cfg['model'])[0]
    sparse_query = embed_sparse(query, get_sparse_options(index_cfg))
    return hybrid_search(dense_vector, sparse_query, collection_name, qdrant_client, limit, query_filter)

def search_and_filter_chunks(query, qdrant_client, embedder, query_filter = None):
    points = search_chunks(query, qdrant_client, embedder, query_filter)
    if not points and query_filter:
        points = search_chunks(query, qdrant_client, embedder)
    return points


if __name__ == '__main__':
    points = search_and_filter_chunks(
        "Quelle était l'HbA1c de M. Jean Simon sur le bilan biologique d'avril 2025 ?",
        get_qdrant_client(), get_embedder()
    )
    for point in points:
        print(round(point.score, 3), point.payload['document'], point.payload['chunk_index'])
