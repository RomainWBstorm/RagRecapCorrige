from rag_recap.config import settings

from qdrant_client import QdrantClient


def get_qdrant_client():
    return QdrantClient(
        url = settings.qdrant_url,
        api_key=settings.qdrant_api_key
    )

def get_collection_name(embedder_name, chunk_type, chunk_size, sparse_vector=None, patient_line=False):
    name = f'{embedder_name}_{chunk_type}_{chunk_size}'
    if patient_line:
        name += '_patient'
    if sparse_vector:
        name += f'_{sparse_vector}'
    return name.replace('/', '-')
