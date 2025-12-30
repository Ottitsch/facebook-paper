"""
Evaluation metrics for RAG models.
Implements Exact Match and F1 score as used in the original RAG paper.
"""

import re
import string
from collections import Counter


def normalize_answer(s):
    """
    Normalize answer text for evaluation.

    Lower case text
    Remove punctuation
    Remove articles (a, an, the)
    Remove extra whitespace
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(char for char in text if char not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    """
    Compute exact match score (0 or 1) between prediction and ground truth.

    Args:
        prediction: Model prediction string
        ground_truth: Ground truth answer string

    Returns:
        1 if normalized strings match exactly, 0 otherwise
    """
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction, ground_truth):
    """
    Compute token-level F1 score between prediction and ground truth.

    Args:
        prediction: Model prediction string
        ground_truth: Ground truth answer string

    Returns:
        F1 score as float between 0 and 1
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        # If either is empty, f1 = 1 if both are empty, 0 otherwise
        return int(prediction_tokens == ground_truth_tokens)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_exact_match(predictions, references):
    """
    Compute Exact Match score over a dataset.

    Args:
        predictions: List of prediction strings
        references: List of reference answers (each can be a list of valid answers)

    Returns:
        Average Exact Match score as percentage (0-100)
    """
    em_scores = []

    for pred, refs in zip(predictions, references):
        # Handle case where refs is a single string
        if isinstance(refs, str):
            refs = [refs]

        # Take max EM across all valid references
        max_em = max(exact_match_score(pred, ref) for ref in refs)
        em_scores.append(max_em)

    return 100.0 * sum(em_scores) / len(em_scores)


def compute_f1(predictions, references):
    """
    Compute F1 score over a dataset.

    Args:
        predictions: List of prediction strings
        references: List of reference answers (each can be a list of valid answers)

    Returns:
        Average F1 score as percentage (0-100)
    """
    f1_scores = []

    for pred, refs in zip(predictions, references):
        # Handle case where refs is a single string
        if isinstance(refs, str):
            refs = [refs]

        # Take max F1 across all valid references
        max_f1 = max(f1_score(pred, ref) for ref in refs)
        f1_scores.append(max_f1)

    return 100.0 * sum(f1_scores) / len(f1_scores)


def compute_metrics(predictions, references):
    """
    Compute all evaluation metrics.

    Args:
        predictions: List of prediction strings
        references: List of reference answers (each can be a list of valid answers)

    Returns:
        Dictionary with 'exact_match' and 'f1' keys
    """
    return {
        'exact_match': compute_exact_match(predictions, references),
        'f1': compute_f1(predictions, references),
        'num_examples': len(predictions)
    }
