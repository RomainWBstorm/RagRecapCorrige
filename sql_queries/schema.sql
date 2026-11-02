CREATE TABLE IF NOT EXISTS run(
    run_id SERIAL CONSTRAINT PK_RUN_ID PRIMARY KEY,
    start_time TIMESTAMPTZ DEFAULT NOW(),
    end_time TIMESTAMPTZ,
    doc_processed INT,
    doc_ignored INT,
    doc_failed INT,
    parameters JSONB
);

CREATE TABLE IF NOT EXISTS document (
    document_id SERIAL CONSTRAINT PK_DOCUMENT_ID PRIMARY KEY,
    run_id INT CONSTRAINT FK_DOCUMENT_RUN REFERENCES run(run_id),
    file_path TEXT,
    export_folder TEXT,
    file_name TEXT,
    file_format_ TEXT,
    file_hash TEXT,
    status_ TEXT,
    page_count INT,
    text_count INT,
    table_count INT,
    image_count INT,
    process_time FLOAT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS experiments (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    config JSONB NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    duration_s REAL,
    status TEXT NOT NULL DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS answers (
    experiment_id INT NOT NULL REFERENCES experiments(id),
    question_id TEXT NOT NULL,
    chunks JSONB NOT NULL,
    answer TEXT,
    latency_ms REAL,
    llm_metrics JSONB,
    PRIMARY KEY (experiment_id, question_id)
);

CREATE TABLE IF NOT EXISTS scores (
    experiment_id INT NOT NULL REFERENCES experiments(id),
    question_id TEXT NOT NULL,
    question_type TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL,
    reason TEXT,
    PRIMARY KEY (experiment_id, question_id, metric)
);