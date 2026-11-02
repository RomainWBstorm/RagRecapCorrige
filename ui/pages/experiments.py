from rag_recap.config import settings
from rag_recap.db import get_connection
from rag_recap.evaluation.metrics import judge_chunk

import json
import polars as pl
import streamlit as st

EXPERIMENTS_QUERY = '''
    SELECT CONCAT(e.id, ' - ', e.name) AS experiment, e.status, COUNT(a.question_id) AS answers,
           ROUND(AVG(a.latency_ms)) AS latency_ms
    FROM experiments e
    LEFT JOIN answers a ON a.experiment_id = e.id
    GROUP BY e.id
    ORDER BY e.id
'''

SCORES_QUERY = '''
    SELECT CONCAT(e.id, ' - ', e.name) AS experiment, s.question_id, s.question_type, s.metric, s.value, s.reason
    FROM scores s
    JOIN experiments e ON e.id = s.experiment_id
'''

ANSWERS_QUERY = '''
    SELECT CONCAT(e.id, ' - ', e.name) AS experiment, a.question_id, a.answer, a.chunks
    FROM answers a
    JOIN experiments e ON e.id = a.experiment_id
    ORDER BY e.id
'''

with get_connection() as conn:
    experiments = pl.read_database(EXPERIMENTS_QUERY, conn)
    scores = pl.read_database(SCORES_QUERY, conn, infer_schema_length=None)
    answers = pl.read_database(ANSWERS_QUERY, conn)

questions = json.loads(settings.eval_dataset_path.read_text(encoding='utf-8'))
question_texts = pl.DataFrame(questions).select(pl.col('id').alias('question_id'), 'question')

st.title('Expériences')

if experiments.is_empty():
    st.info('Aucune expérience en base : lancez rag_recap.evaluation.runner.')
    st.stop()

metric = st.selectbox('Métrique', scores['metric'].unique().sort())

st.header('Classement')
ranking = (
    scores.filter(pl.col('metric') == metric)
    .group_by('experiment')
    .agg(pl.col('value').mean().round(3).alias('mean'), pl.col('value').count().alias('n'))
    .join(experiments, on='experiment')
    .sort('mean', descending=True)
)
st.dataframe(ranking)

st.header('Graphiques')
st.subheader('Moyenne par métrique')
st.caption('refusal est exclu : sa lecture dépend du type de question, voir le graphique suivant.')
means = (
    scores.filter(pl.col('metric') != 'refusal')
    .group_by('experiment', 'metric')
    .agg(pl.col('value').mean().alias('mean'))
    .sort('metric', 'experiment')
)
st.bar_chart(means, x='metric', y='mean', color='experiment', stack=False)

st.subheader(f'Moyenne de {metric} par type de question')
means_by_type = (
    scores.filter(pl.col('metric') == metric)
    .group_by('experiment', 'question_type')
    .agg(pl.col('value').mean().alias('mean'))
    .sort('question_type', 'experiment')
)
st.bar_chart(means_by_type, x='question_type', y='mean', color='experiment', stack=False, horizontal=True)

st.header('Comparaison de deux expériences')
col1, col2 = st.columns(2)
experiment_a = col1.selectbox('Expérience A', experiments['experiment'])
experiment_b = col2.selectbox('Expérience B', experiments['experiment'], index=experiments.height - 1)

scores_a = scores.filter(pl.col('experiment') == experiment_a, pl.col('metric') == metric)
scores_b = scores.filter(pl.col('experiment') == experiment_b, pl.col('metric') == metric)
comparison = (
    scores_a.select('question_id', 'question_type', pl.col('value').alias('A'))
    .join(scores_b.select('question_id', pl.col('value').alias('B')), on='question_id')
    .with_columns((pl.col('B') - pl.col('A')).alias('gain'))
    .join(question_texts, on='question_id')
)
if metric == 'refusal':
    st.caption('refusal vaut 1 quand le modèle refuse de répondre. Refuser plus est un gain pour out_of_corpus, une perte pour les autres types.')
    comparison = comparison.with_columns(
        pl.when(pl.col('question_type') == 'out_of_corpus')
        .then(pl.col('gain'))
        .otherwise(-pl.col('gain'))
        .alias('gain')
    )
st.caption(f'{comparison.height} questions évaluées par les deux expériences')

st.subheader('Moyennes par type de question')
by_type = (
    comparison.group_by('question_type')
    .agg(pl.len().alias('n'), pl.col('A').mean().round(3), pl.col('B').mean().round(3))
    .sort('question_type')
)
st.dataframe(by_type)

st.subheader('Questions améliorées')
st.dataframe(comparison.filter(pl.col('gain') > 0).sort('gain', descending=True))

st.subheader('Questions dégradées')
st.dataframe(comparison.filter(pl.col('gain') < 0).sort('gain'))

st.header("Détail d'une question")
question = st.selectbox('Question', questions, format_func=lambda q: f"{q['id']} ({q['type']}) {q['question']}")
st.markdown(f"**Question :** {question['question']}")
st.markdown(f"**Réponse attendue :** {question['answer']}")

question_scores = scores.filter(pl.col('question_id') == question['id'])
for answer in answers.filter(pl.col('question_id') == question['id']).iter_rows(named=True):
    st.subheader(answer['experiment'])
    st.markdown(f"**Réponse :** {answer['answer']}")

    chunks = []
    for chunk in answer['chunks']:
        relevant = any(judge_chunk(chunk, evidence) for evidence in question['evidence'])
        chunks.append({'document': chunk['document'], 'chunk_index': chunk['chunk_index'],
                       'score': round(chunk['score'], 3), 'relevant': relevant})
    st.dataframe(chunks)

    st.table(question_scores.filter(pl.col('experiment') == answer['experiment']).select('metric', 'value', 'reason'))
