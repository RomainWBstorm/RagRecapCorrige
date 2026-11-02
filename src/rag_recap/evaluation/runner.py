from rag_recap.config import settings, runner_settings
from rag_recap.db import start_experiment, end_experiment, insert_answer, insert_scores, get_answers
from rag_recap.qdrant import get_qdrant_client
from rag_recap.embedding import get_embedder
from rag_recap.rag.ask import generate_response, get_llm_client
from rag_recap.evaluation.metrics import retrieval_scores, is_refusal
from rag_recap.evaluation.judge import get_judge_metrics, get_judge_sample, judge_scores

from openai import OpenAI
from pathlib import Path
from time import perf_counter

import asyncio
import json
import sys


def load_questions(path):
    with open(path, encoding = 'utf-8') as f:
        return json.load(f)

def get_experiment_config():
    generation_cfg = runner_settings['generation']
    system_prompt = Path(generation_cfg['system_prompt']).read_text(encoding = 'utf-8')
    return {**runner_settings, 'generation': {**generation_cfg, 'system_prompt_text': system_prompt}}

def evaluate_question(question, answer, chunks):
    scores = [('refusal', float(is_refusal(answer)), None)]
    if question['evidence']:
        scores += retrieval_scores(question, chunks, runner_settings['reranker']['top_k'])
    return scores

async def answer_question(experiment_id, question, qdrant_client, embedder, llm_client, semaphore):
    async with semaphore:
        begin = perf_counter()
        answer, llm_metrics, reranked_chunks = await generate_response(question['question'], qdrant_client, embedder, llm_client)
        latency_ms = (perf_counter() - begin) * 1000

    chunks = [{**chunk.payload, 'score': score} for chunk, score in reranked_chunks]
    insert_answer(experiment_id, question['id'], chunks, answer, latency_ms, llm_metrics)
    insert_scores(experiment_id, question['id'], question['type'], evaluate_question(question, answer, chunks))
    print(question['id'], question['type'], f'{latency_ms:.0f} ms')

async def answer_questions(experiment_id, questions):
    qdrant_client = get_qdrant_client()
    embedder = get_embedder()
    llm_client = get_llm_client()
    semaphore = asyncio.Semaphore(settings.llm_parallel_requests)

    await asyncio.gather(*[
        answer_question(experiment_id, question, qdrant_client, embedder, llm_client, semaphore)
        for question in questions
    ])

def run_experiment():
    questions = load_questions(settings.eval_dataset_path)
    experiment_id = start_experiment(runner_settings['name'], get_experiment_config())
    print('Experiment', experiment_id)
    start = perf_counter()
    status = 'failed'

    try:
        asyncio.run(answer_questions(experiment_id, questions))
        status = 'done'
    finally:
        end_experiment(experiment_id, perf_counter() - start, status)

def get_served_models():
    llm_client = OpenAI(base_url = settings.llm_url, api_key = settings.llm_server_api_key)
    return [model.id for model in llm_client.models.list()]

def judge_experiment(experiment_id):
    judge_model = runner_settings['judge']['model']
    if judge_model not in get_served_models():
        raise ValueError(f'The judge model {judge_model} is not served by the LLM server')

    questions = load_questions(settings.eval_dataset_path)
    answers = get_answers(experiment_id)
    judge_metrics = get_judge_metrics()
    judge_sample = get_judge_sample(questions)

    for question in questions:
        if question['id'] in judge_sample:
            answer, chunks = answers[question['id']]
            insert_scores(experiment_id, question['id'], question['type'], judge_scores(question, answer, chunks, judge_metrics))
            print(question['id'], question['type'], 'judged')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'judge':
        for experiment_id in sys.argv[2:]:
            judge_experiment(int(experiment_id))
    else:
        run_experiment()
