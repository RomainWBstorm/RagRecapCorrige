from rag_recap.db import get_connection

import polars as pl
import streamlit as st

RUNS_QUERY = '''
    SELECT r.run_id, r.start_time, ROUND(EXTRACT(EPOCH FROM r.end_time - r.start_time), 1) AS duration_s,
           r.doc_processed, r.doc_ignored, COUNT(d.error) AS errors, r.parameters::text AS parameters
    FROM run r
    LEFT JOIN document d ON d.run_id = r.run_id
    GROUP BY r.run_id
    ORDER BY r.run_id
'''

DOCUMENTS_QUERY = '''
    SELECT document_id, run_id, file_name, file_format_, status_, page_count,
           text_count, table_count, image_count, process_time, error
    FROM document
    ORDER BY document_id
'''

with get_connection() as conn:
    runs = pl.read_database(RUNS_QUERY, conn)
    documents = pl.read_database(DOCUMENTS_QUERY, conn)

errors = documents.filter(pl.col('error').is_not_null())

st.title('Logs de conversion')

col1, col2, col3 = st.columns(3)
col1.metric('Runs', runs.height)
col2.metric('Documents', documents.height)
col3.metric('Erreurs', errors.height)

st.header('Runs')
st.dataframe(runs)

st.header('Documents')

col1, col2, col3 = st.columns(3)
run_ids = col1.multiselect('Run', documents['run_id'].unique().sort(), placeholder='Tous')
statuses = col2.multiselect('Statut', documents['status_'].unique().sort(), placeholder='Tous')
formats = col3.multiselect('Format', documents['file_format_'].unique().sort(), placeholder='Tous')

selection = documents
if run_ids:
    selection = selection.filter(pl.col('run_id').is_in(run_ids))
if statuses:
    selection = selection.filter(pl.col('status_').is_in(statuses))
if formats:
    selection = selection.filter(pl.col('file_format_').is_in(formats))

st.caption(f'{selection.height} documents')
st.dataframe(selection)

st.subheader('Temps de traitement moyen par format')
time_by_format = selection.group_by('file_format_').agg(pl.col('process_time').mean())
st.bar_chart(time_by_format, x='file_format_', y='process_time', x_label='Format', y_label='Secondes')

st.header('Erreurs')
errors_by_message = (
    errors.group_by('error')
    .agg(pl.len().alias('count'), pl.col('file_name').sort().alias('files'))
    .sort('count', descending=True)
)
st.dataframe(errors_by_message)
