"""
Prompt templates for RAG experiments
"""


PROMPT_TEMPLATES = {
    "standard": {
        "template": "question: {question} context: {context}",
        "description": "Minimal format: question + context concatenation"
    },

    "reasoning": {
        "template": (
            "Answer the question based on the context below. Explain your reasoning step by step.\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Reasoning & Answer:"
        ),
        "description": "Multi-step reasoning with explicit step-by-step instruction"
    },

    "instruction": {
        "template": (
            "You are a helpful assistant. Use the provided context to answer the user question "
            "accurately and concisely and faithfully.\n\n"
            "Context: {context}\n\n"
            "User: {question}\n"
            "Assistant:"
        ),
        "description": "Role-based format with persona and faithfulness instruction"
    },

    "context_first": {
        "template": (
            "Background information:\n{context}\n\n"
            "Based on the above context, answer the following question:\n"
            "{question}\n\n"
            "Answer:"
        ),
        "description": "Context-first ordering for better context integration"
    },

    "qa_format": {
        "template": (
            "CONTEXT:\n{context}\n\n"
            "QUESTION:\n{question}\n\n"
            "ANSWER:"
        ),
        "description": "Explicit QA format with labeled sections"
    },

    "retrieval_aware": {
        "template": (
            "question: {question}\n"
            "passages:\n{context}\n\n"
            "Based on these passages, answer: {question}"
        ),
        "description": "Includes passage labels to help model distinguish sources"
    },

    "multi_turn": {
        "template": (
            "User: Please answer the following question based on the context provided.\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Assistant:"
        ),
        "description": "Multi-turn conversational format"
    },

    "summary_based": {
        "template": (
            "First, summarize the key information in the context below.\n"
            "Then, use that summary to answer the question.\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Summary and Answer:"
        ),
        "description": "Prompts model to first summarize context, then answer"
    },
}


def format_prompt(question, documents, prompt_type="standard"):
    """
    Format a prompt using specified template.

    Args:
        question: Question string
        documents: List of document dicts with 'text' field
        prompt_type: Type of prompt template to use

    Returns:
        Formatted prompt string
    """
    if prompt_type not in PROMPT_TEMPLATES:
        raise ValueError(
            f"Unknown prompt type: {prompt_type}. "
            f"Available: {', '.join(PROMPT_TEMPLATES.keys())}"
        )

    # Extract and concatenate context
    context_parts = []
    for doc in documents:
        text = doc.get('text', '')
        if text:
            context_parts.append(text)

    context = " ".join(context_parts)

    # Get template and format
    template = PROMPT_TEMPLATES[prompt_type]["template"]
    prompt = template.format(question=question, context=context)

    return prompt


def get_prompt_info(prompt_type):
    """
    Get metadata about a prompt type.

    Args:
        prompt_type: Name of prompt template

    Returns:
        Dictionary with description and template
    """
    if prompt_type not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    return PROMPT_TEMPLATES[prompt_type].copy()


def get_available_prompts():
    """
    Get list of available prompt types.

    Returns:
        List of prompt type names
    """
    return list(PROMPT_TEMPLATES.keys())
