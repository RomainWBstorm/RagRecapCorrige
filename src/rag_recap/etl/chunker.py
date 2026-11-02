from rag_recap.config import settings, runner_settings

from transformers import AutoTokenizer
from docling_core.types.doc import DoclingDocument
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

from pathlib import Path

import json

CHUNKERS = {
    'hybrid': HybridChunker,
    'hierarchical': HierarchicalChunker
}

def get_chunker(chunker_type, embedder_name, max_tokens):
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained((Path(settings.models_folder) / 'embedder' / embedder_name)),
        max_tokens=max_tokens
    )
    chunker = CHUNKERS.get(chunker_type)(tokenizer=tokenizer)
    return chunker

def chunk_document(document_path, chunker):
    document = DoclingDocument.load_from_json(document_path)
    
    origin_file_path = Path(document.origin.filename)

    origin_file_name = origin_file_path.stem
    origin_file_extension = origin_file_path.suffix.lstrip('.').lower() 

    for i, chunk in enumerate(chunker.chunk(dl_doc=document)):
        yield {
            'document': origin_file_name,
            'chunk_index': i,
            'headings': ' | '.join(chunk.meta.headings or []),
            'text': chunker.contextualize(chunk),
            'file_extension': origin_file_extension,
        }

def load_documents_metadata(path):
    with open(path, encoding = 'utf-8') as f:
        return {Path(document['document']).stem: document for document in json.load(f)}

def get_patient_line(metadata):
    if metadata['patient_name'] is None:
        return ''
    return f"Patient : {metadata['patient_name']} (IPP {metadata['patient_ipp']}) - {metadata['title']} du {metadata['date']}\n"

def chunk_document_with_patient_line(document_path, chunker, documents_metadata):
    for chunk in chunk_document(document_path, chunker):
        metadata = documents_metadata[chunk['document']]
        yield {
            **chunk,
            'text': get_patient_line(metadata) + chunk['text'],
            'patient_ipp': metadata['patient_ipp'],
            'patient_name': metadata['patient_name'],
            'doc_type': metadata['doc_type'],
            'date': metadata['date'],
        }

def iter_chunks(documents_folder, chunker):
    patient_line = runner_settings['chunking']['patient_line']
    documents_metadata = load_documents_metadata(settings.documents_metadata_path)
    for document_path in Path(documents_folder).rglob('*.json'):
        if patient_line:
            yield from chunk_document_with_patient_line(document_path, chunker, documents_metadata)
        else:
            yield from chunk_document(document_path, chunker)

if  __name__ == '__main__':
    pass