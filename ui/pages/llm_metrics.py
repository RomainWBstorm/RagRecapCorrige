from rag_recap.config import settings
from rag_recap.db import get_connection

import json
import polars as pl
import streamlit as st

METRICS_QUERY = '''
    SELECT CONCAT(e.id, ' - ', e.name) AS experiment, a.question_id, a.latency_ms,
           (a.llm_metrics->>'prompt_n')::INT AS prompt_tokens,
           (a.llm_metrics->>'cache_n')::INT AS cached_tokens,
           (a.llm_metrics->>'predicted_n')::INT AS completion_tokens,
           (a.llm_metrics->>'prompt_ms')::REAL AS prompt_ms,
           (a.llm_metrics->>'predicted_ms')::REAL AS predicted_ms,
           (a.llm_metrics->>'prompt_per_second')::REAL AS prompt_per_second,
           (a.llm_metrics->>'predicted_per_second')::REAL AS predicted_per_second,
           (a.llm_metrics->>'draft_n')::INT AS draft_tokens,
           (a.llm_metrics->>'draft_n_accepted')::INT AS draft_accepted
    FROM answers a
    JOIN experiments e ON e.id = a.experiment_id
    WHERE a.llm_metrics IS NOT NULL
    ORDER BY e.id
'''

with get_connection() as conn:
    metrics = pl.read_database(METRICS_QUERY, conn)

st.title('Performances LLM')
st.caption('Métriques renvoyées par llama-server pour chaque réponse générée (objet timings).')

if metrics.is_empty():
    st.info('Aucune réponse avec des métriques llama-server : lancez une expérience.')
    st.stop()

questions = json.loads(settings.eval_dataset_path.read_text(encoding='utf-8'))
question_types = pl.DataFrame(questions).select(pl.col('id').alias('question_id'), pl.col('type').alias('question_type'))
metrics = metrics.join(question_types, on='question_id')

st.header('Résumé par expérience')
summary = (
    metrics.group_by('experiment')
    .agg(
        pl.len().alias('answers'),
        pl.col('latency_ms').mean().round(0),
        pl.col('prompt_tokens').mean().round(0),
        pl.col('completion_tokens').mean().round(0),
        pl.col('prompt_per_second').mean().round(1),
        pl.col('predicted_per_second').mean().round(1),
        (pl.col('cached_tokens').sum() / (pl.col('cached_tokens') + pl.col('prompt_tokens')).sum()).round(3).alias('cache_ratio'),
        (pl.col('draft_accepted').sum() / pl.col('draft_tokens').sum()).round(3).alias('draft_acceptance'),
    )
    .sort('experiment')
)
st.dataframe(summary)
st.caption('cache_ratio : part des tokens du prompt déjà en cache. draft_acceptance : part des tokens proposés par le décodage spéculatif qui ont été acceptés.')

st.header('Graphiques')
st.subheader('Vitesse de génération moyenne (tokens/s)')
st.bar_chart(summary, x='experiment', y='predicted_per_second')

st.subheader('Latence en fonction du nombre de tokens générés')
st.scatter_chart(metrics, x='completion_tokens', y='latency_ms', color='experiment')

st.subheader('Latence moyenne par type de question (ms)')
latency_by_type = (
    metrics.group_by('experiment', 'question_type')
    .agg(pl.col('latency_ms').mean().round(0))
    .sort('question_type', 'experiment')
)
st.bar_chart(latency_by_type, x='question_type', y='latency_ms', color='experiment', stack=False, horizontal=True)

st.header('Détail par question')
st.dataframe(metrics.sort('latency_ms', descending=True))
