import numpy as np

from rag_recap.config import settings, runner_settings
from rag_recap.etl.convert import convert_corpus
from rag_recap.etl.chunker import iter_chunks, get_chunker
from rag_recap.embedding import get_embedder, get_sparse_options, embed, embed_sparse
from rag_recap.qdrant import get_qdrant_client, get_collection_name

from docling_core.types.doc import DoclingDocument

from qdrant_client import models

import uuid
from itertools import batched


def init_collection(qdrant_client, collection_name, vector_size):
    if qdrant_client.collection_exists(collection_name):
        return

    qdrant_client.create_collection(
        collection_name,
        vectors_config = {
            'dense': models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        },
        sparse_vectors_config = {
            'sparse': models.SparseVectorParams(modifier=models.Modifier.IDF),
        },
    )

    qdrant_client.create_payload_index(
        collection_name,
        field_name='document',
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

def get_chunk_uuid(chunk):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk['document']}/{chunk['chunk_index']}"))

def iter_points(document_folder, chunker, embedder, model_name, embedding_batch_size, sparse_embedder_options):
    chunks = iter_chunks(document_folder, chunker)
    for batch in batched(chunks, embedding_batch_size):
        dense_vectors = embed([chunk['text'] for chunk in batch], embedder, model_name)
        for chunk, dense_vector in zip(batch, dense_vectors):
            yield models.PointStruct(
                id=get_chunk_uuid(chunk),
                vector = {
                    'dense': dense_vector,
                    'sparse': embed_sparse(chunk['text'], sparse_embedder_options)
                }
                , payload = chunk
            )

def probe_avg_len(document_folder, chunker, probe_size: int = 500):
    chunks = iter_chunks(document_folder, chunker)
    for batch in batched(chunks, probe_size):
        return int(np.mean([len(chunk['text'].split()) for chunk in batch]))

def index_chunks(qdrant_client, collection_name, document_folder, chunker, embedder, model_name, embedding_batch_size, sparse_embedder_options):    
    points = iter_points(document_folder, chunker, embedder, model_name, embedding_batch_size, sparse_embedder_options)
    qdrant_client.upload_points(collection_name, points=points, batch_size=64)

def index_corpus():
    embedder_cfg = runner_settings['embedder']
    chunking_cfg = runner_settings['chunking']
    index_cfg = runner_settings['index']

    embedder = get_embedder()
    chunker = get_chunker(chunking_cfg['type'], embedder_cfg['model'], chunking_cfg['max_tokens'])
    qdrant_client = get_qdrant_client()

    collection_name = get_collection_name(
        embedder_name = embedder_cfg['model'],
        chunk_type = chunking_cfg['type'],
        chunk_size = chunking_cfg['max_tokens'],
        sparse_vector = index_cfg.get('sparse_vector'),
        patient_line = chunking_cfg['patient_line'],
    )

    init_collection(qdrant_client, collection_name, embedder_cfg['vector_size'])

    convert_corpus()

    sparse_embedder_options = get_sparse_options(index_cfg)
    sparse_embedder_options['avg_len'] = probe_avg_len(settings.processed_documents_folder, chunker)

    index_chunks(
        qdrant_client, collection_name , settings.processed_documents_folder, chunker,
        embedder, embedder_cfg['model'], embedder_cfg['embedding_batch_size'], sparse_embedder_options
        )


if __name__ == '__main__':
    index_corpus()

