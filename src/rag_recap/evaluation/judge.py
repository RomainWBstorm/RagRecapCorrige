from rag_recap.config import settings, runner_settings

from deepeval.models import LocalModel
from deepeval.metrics import FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams


def get_judge_model():
    return LocalModel(
        model = runner_settings['judge']['model'],
        base_url = settings.llm_url,
        api_key = settings.llm_server_api_key,
        temperature = 0,
        generation_kwargs = {'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}}
    )

def get_judge_metrics():
    judge_model = get_judge_model()
    return {
        'faithfulness': FaithfulnessMetric(model = judge_model, async_mode = False),
        'correctness': GEval(
            name = 'correctness',
            criteria = 'Determine whether the actual output is factually correct and complete compared to the expected output.',
            evaluation_params = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
            model = judge_model, async_mode = False
        ),
    }

def get_judge_sample(questions, per_type = 2):
    sample_ids = []
    for question_type in sorted({question['type'] for question in questions} - {'out_of_corpus'}):
        sample_ids += [question['id'] for question in questions if question['type'] == question_type][:per_type]
    return sample_ids

def judge_scores(question, answer, chunks, metrics):
    test_case = LLMTestCase(
        input = question['question'],
        actual_output = answer,
        expected_output = question['answer'],
        retrieval_context = [chunk['text'] for chunk in chunks],
    )

    scores = []
    for name, metric in metrics.items():
        try:
            metric.measure(test_case)
            scores.append((name, metric.score, metric.reason))
        except Exception as e:
            scores.append((name, None, f'{type(e).__name__}: {e}'))
    return scores
