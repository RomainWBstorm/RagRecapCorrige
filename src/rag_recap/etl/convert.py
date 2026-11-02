from rag_recap.config import settings
from rag_recap.db import start_run, end_run, insert_log, is_file_processed

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions

from time import perf_counter

import hashlib

ALLOWED_INPUT_FORMAT = [InputFormat.PDF, InputFormat.DOC, InputFormat.DOCX]
ALLOWED_SUFFIXES = {".pdf", ".docx", ".doc"}
TRUNCATE_EXPORT_FOLDER = False

def get_converter(models_folder, device, num_threads):
    options = PdfPipelineOptions(
        artifacts_path = (models_folder / 'docling'),
        accelerator_options = AcceleratorOptions(
            device = device,
            num_threads = num_threads
        ),
        generate_page_images = True
    )

    converter = DocumentConverter(
        allowed_formats = ALLOWED_INPUT_FORMAT,
        format_options= {
            InputFormat.PDF: PdfFormatOption(pipeline_options=options)
        }
    )

    return converter

def list_allowed_files(corpus_folder):
    return sorted(
        f for f in corpus_folder.rglob('*') if f.is_file() and f.suffix.lower() in ALLOWED_SUFFIXES
        )

def get_hash(file_path):
    return hashlib.sha256(file_path.read_bytes()).hexdigest()

def convert_file(file_path, converter, export_folder, file_hash):
    log = {
        'file_path': str(file_path.absolute().as_posix()),
        'export_folder': (export_folder.absolute().as_posix()),
        'file_name': file_path.name,
        'file_format_': file_path.suffix.lstrip('.'),
        'file_hash': file_hash,
        'status_': None,
        'page_count': None,
        'text_count': None,
        'table_count': None,
        'image_count': None,
        'process_time': None,
        'error': None
    }

    start = perf_counter()

    try:
        result = converter.convert(file_path)
        doc = result.document
        doc.save_as_json(export_folder / f'{file_hash}.json')
        doc.save_as_markdown(export_folder / f'{file_hash}.md')
        log.update(
            status_ = result.status.value,
            page_count = doc.num_pages(),
            text_count = len(doc.texts),
            table_count = len(doc.tables),
            image_count =  len(doc.pictures),
        )
    except Exception as e:
        log.update(status_='failure', error=f'{type(e).__name__}: {e}')
    finally:
        log.update(process_time = perf_counter() - start)

    return log

def convert_corpus():
    converter = get_converter(
        settings.models_folder,
        settings.ocr_device,
        settings.ocr_num_threads,
    )

    conversion_parameters = {
        'device': settings.ocr_device,
        'threads': settings.ocr_num_threads,
        'allowed_suffix': ','.join(ALLOWED_SUFFIXES)
    }

    run_id = start_run(conversion_parameters)

    files = list_allowed_files(settings.corpus_folder)
    
    settings.processed_documents_folder.mkdir(exist_ok = True)

    run_logs = []

    doc_processed = doc_ignored = doc_failed = 0

    try:
        for file_path in files:
            file_hash = get_hash(file_path)
            if not is_file_processed(file_path, file_hash):
                log = convert_file(file_path, converter, settings.processed_documents_folder, file_hash)
                doc_processed += 1

                insert_log(run_id, log)

                run_logs.append(log)

                if log.get('status_') == 'failure':
                    doc_failed += 1
            else:
                doc_ignored += 1
    finally:
        end_run(run_id, doc_processed, doc_ignored, doc_failed)

if __name__ == '__main__':
    convert_corpus()