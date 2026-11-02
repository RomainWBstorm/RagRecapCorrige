from pathlib import Path

from ranx import Qrels, Run, evaluate

REFUSAL = 'je ne dispose pas de cette information'


def judge_chunk(chunk, evidence):
    return chunk['document'] == Path(evidence['document']).stem and all(snippet in chunk['text'] for snippet in evidence['snippets'])

def get_evidence_id(evidence, i):
    if evidence['fact'] is None:
        return f'extra_{i}'
    return f"fact_{evidence['fact']}"

def build_qrels(question):
    return {get_evidence_id(evidence, i): evidence['grade'] for i, evidence in enumerate(question['evidence'])}

def build_run(question, chunks):
    run = {}
    for rank, chunk in enumerate(chunks):
        score = float(len(chunks) - rank)
        evidence_ids = [get_evidence_id(evidence, i) for i, evidence in enumerate(question['evidence']) if judge_chunk(chunk, evidence)]
        for evidence_id in evidence_ids or [f'chunk_{rank}']:
            run.setdefault(evidence_id, score)
    return run

def retrieval_scores(question, chunks, k):
    metrics = {
        f'hit@{k}': f'hit_rate@{k}-l2',
        'mrr': 'mrr-l2',
        f'recall@{k}': f'recall@{k}-l2',
        f'ndcg@{k}': f'ndcg@{k}',
    }
    qrels = Qrels({question['id']: build_qrels(question)})
    run = Run({question['id']: build_run(question, chunks)})
    values = evaluate(qrels, run, list(metrics.values()))
    return [(name, float(values[ranx_name]), None) for name, ranx_name in metrics.items()]

def is_refusal(answer):
    return REFUSAL in answer.lower()
