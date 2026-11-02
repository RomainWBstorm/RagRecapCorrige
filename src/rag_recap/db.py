from rag_recap.config import settings
import psycopg as pg
from psycopg.types.json import Jsonb

def init_db():
    with get_connection() as conn:
        conn.execute(load_sql('schema.sql'))

def load_sql(name):
    return (settings.sql_statement_folder / f'{name}').read_text('utf-8')
    
def get_connection(autocommit = True):
    return pg.connect(settings.postgres_url, autocommit=autocommit)

def start_run(parameters):
    init_db()
    with get_connection() as conn:
        return conn.execute('INSERT INTO run(parameters) VALUES(%s) RETURNING run_id', (Jsonb(parameters),)).fetchone()[0]

def end_run(run_id, doc_processed, doc_ignored, doc_failed):
    with get_connection() as conn:
        conn.execute(
            'UPDATE run SET end_time = NOW(), doc_processed = %s, doc_ignored = %s, doc_failed = %s WHERE run_id = %s',
            (doc_processed, doc_ignored, doc_failed, run_id)
        )

def insert_log(run_id, log):
    with get_connection() as conn:
        conn.execute(
            '''INSERT INTO document (run_id, file_path, export_folder, file_name, file_format_, file_hash
                                    , status_, page_count, text_count, table_count, image_count, process_time, error)
            VALUES (%(run_id)s, %(file_path)s, %(export_folder)s, %(file_name)s, %(file_format_)s, %(file_hash)s,
                    %(status_)s, %(page_count)s, %(text_count)s, %(table_count)s, %(image_count)s, %(process_time)s, %(error)s)''',
            {'run_id': run_id, **log},
        )

def start_experiment(name, config):
    init_db()
    with get_connection() as conn:
        return conn.execute('INSERT INTO experiments(name, config) VALUES(%s, %s) RETURNING id', (name, Jsonb(config))).fetchone()[0]

def end_experiment(experiment_id, duration_s, status):
    with get_connection() as conn:
        conn.execute(
            'UPDATE experiments SET duration_s = %s, status = %s WHERE id = %s',
            (duration_s, status, experiment_id)
        )

def insert_answer(experiment_id, question_id, chunks, answer, latency_ms, llm_metrics):
    with get_connection() as conn:
        conn.execute(
            'INSERT INTO answers (experiment_id, question_id, chunks, answer, latency_ms, llm_metrics) VALUES (%s, %s, %s, %s, %s, %s)',
            (experiment_id, question_id, Jsonb(chunks), answer, latency_ms, Jsonb(llm_metrics))
        )

def get_answers(experiment_id):
    with get_connection() as conn:
        rows = conn.execute('SELECT question_id, answer, chunks FROM answers WHERE experiment_id = %s', (experiment_id,)).fetchall()
    return {question_id: (answer, chunks) for question_id, answer, chunks in rows}

def insert_scores(experiment_id, question_id, question_type, scores):
    with get_connection() as conn:
        conn.cursor().executemany(
            'INSERT INTO scores (experiment_id, question_id, question_type, metric, value, reason) VALUES (%s, %s, %s, %s, %s, %s)',
            [(experiment_id, question_id, question_type, metric, value, reason) for metric, value, reason in scores]
        )

def is_file_processed(file_path, file_hash):
    with get_connection() as conn:
        r = conn.execute(
            "SELECT 1 FROM document WHERE file_hash = (%s) AND status_ IN ('success', 'partial_success') LIMIT 1",
            (file_hash,)
        ).fetchone()
    return r is not None