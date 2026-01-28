import logging
import re
import string
from collections import Counter
from typing import List

logger = logging.getLogger(__name__)


# =============================================================================
# Metrics (HotPotQA official evaluation)
# =============================================================================

def normalize_answer(s: str) -> str:
    """Normalize answer for evaluation (official HotPotQA normalization)."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def calculate_f1(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """Calculate F1, Precision, Recall (official HotPotQA implementation).

    Returns:
        Tuple of (f1, precision, recall)
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0.0, 0.0, 0.0)

    # Special handling for yes/no answers
    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return ZERO_METRIC

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def calculate_exact_match(prediction: str, ground_truth: str) -> bool:
    """Calculate exact match (official HotPotQA implementation)."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


# Stopwords for adherence calculation
STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'to', 'of', 'in', 'for', 'on', 'with',
    'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once',
    'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'not', 'only', 'same', 'than',
    'too', 'very', 'just', 'also', 'now', 'it', 'its', 'this', 'that',
    'these', 'those', 'i', 'you', 'he', 'she', 'they', 'we', 'what', 'which',
    'who', 'whom', 'how', 'when', 'where', 'why', 'all', 'any', 'if'
}


def calculate_context_relevance(retrieved_docs: List[str], gold_docs: List[str]) -> float:
    """Calculate how many retrieved docs are in the gold set."""
    if not retrieved_docs or not gold_docs:
        return 0.0
    retrieved_set = set(doc.strip().lower() for doc in retrieved_docs)
    gold_set = set(doc.strip().lower() for doc in gold_docs)
    overlap = len(retrieved_set & gold_set)
    return overlap / len(retrieved_set) if retrieved_set else 0.0


def calculate_context_utilization(answer: str, retrieved_docs: List[str]) -> float:
    """Calculate how much of the retrieved docs are used in the answer."""
    if not answer or not retrieved_docs:
        return 0.0
    answer_tokens = set(answer.lower().split())
    docs_used = sum(1 for doc in retrieved_docs if len(answer_tokens & set(doc.lower().split())) > 3)
    return docs_used / len(retrieved_docs)


def calculate_adherence(answer: str, retrieved_docs: List[str]) -> float:
    """Calculate if the answer is grounded in the retrieved docs."""
    if not answer or not retrieved_docs:
        return 0.0

    answer_tokens = set(answer.lower().split()) - STOPWORDS
    if not answer_tokens:
        return 1.0

    all_doc_tokens = set()
    for doc in retrieved_docs:
        all_doc_tokens.update(doc.lower().split())

    grounded_tokens = answer_tokens & all_doc_tokens
    return len(grounded_tokens) / len(answer_tokens)


def llm_judge_correctness(predicted: str, ground_truth: str, question: str, llm_client, model: str) -> dict:
    """Use an LLM to judge if the predicted answer is correct."""
    judge_prompt = f"""You are an expert evaluator judging the correctness of question-answering systems.

    Question: {question}
    Ground Truth Answer: {ground_truth}
    Predicted Answer: {predicted}

    Evaluate if the predicted answer is semantically equivalent to the ground truth answer.
    Respond in this exact format:
        CORRECT: [yes/no]
        CONFIDENCE: [0.0-1.0]
        REASONING: [brief explanation]"""

    try:
        response = llm_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a precise answer evaluator."},
                {"role": "user", "content": judge_prompt}
            ],
            model=model, temperature=0, max_tokens=200
        )
        content = response.choices[0].message.content.strip()

        is_correct, confidence, reasoning = False, 0.5, ""
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith("CORRECT:"):
                is_correct = "yes" in line.lower()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":")[-1].strip())
                except:
                    confidence = 1.0 if is_correct else 0.0
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[-1].strip()

        return {
            "score": confidence if is_correct else (1.0 - confidence),
            "is_correct": is_correct,
            "confidence": confidence,
            "reasoning": reasoning
        }
    except Exception as e:
        logger.error(f"LLM judge error: {e}")
        return {"score": 0.0, "is_correct": False, "confidence": 0.0, "reasoning": str(e)}
