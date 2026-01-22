"""
Unified Reranker for RAG with multiple strategies.
Supports different reranking strategies via parameter:
  - strategy="basic": Pure cross-encoder scoring (baseline)
  - strategy="enhanced": Ensemble of cross-encoder + TF-IDF + term coverage (improved)
"""

import re
from typing import Dict, List

import numpy as np
import torch


class Reranker:
    """
    Unified Reranker with multiple strategies.
    
    Strategies:
    - "basic": Pure cross-encoder scoring (MS-MARCO model)
    - "enhanced": Multi-factor ensemble
      * 50% Cross-Encoder Score (semantic matching)
      * 30% TF-IDF Overlap (exact/partial match)
      * 20% Query Term Coverage (how many query terms in document)
    - "diversity": Semantic relevance + diversity-aware reranking
      * 60% Cross-Encoder Score (semantic matching)
      * 40% Diversity penalty (avoid similar/redundant results)
    """

    def __init__(self, strategy="enhanced", use_fp16=True):
        """
        Initialize reranker with chosen strategy.

        Args:
            strategy: "basic", "enhanced", or "diversity" (default: "enhanced")
            use_fp16: Use FP16 precision if available

        Returns:
            None
        """
        if strategy not in ["basic", "enhanced", "diversity"]:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'basic', 'enhanced', or 'diversity'.")
        
        self.strategy = strategy
        self.use_fp16 = use_fp16
        
        from sentence_transformers import CrossEncoder

        print(f"\nLoading Reranker with strategy: {strategy.upper()}")
        print("="*60)

        # Select model based on strategy
        if strategy == "basic":
            model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
            print(f"Model: {model_name}")
            print(f"Strategy: Pure cross-encoder scoring")
        elif strategy == "enhanced":
            model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
            print(f"Model: {model_name}")
            print(f"Strategy: Multi-factor (50% CE + 30% TF-IDF + 20% Coverage)")
        else:  # diversity
            model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
            print(f"Model: {model_name}")
            print(f"Strategy: Semantic + Diversity (60% CE + 40% Anti-redundancy)")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")

        self.model = CrossEncoder(model_name, device=device, num_labels=1)
        self.device = device
        self.model_name = model_name
        self.use_fp16 = use_fp16 and device == "cuda"

        # Set to eval mode
        self.model.eval()

        model_size = sum(p.numel() for p in self.model.model.parameters()) / 1e6
        print(f"Model size: {model_size:.1f}M parameters")
        print("="*60 + "\n")

    def _normalize_text(self, text):
        """Normalize text for matching."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return set(text.split())

    def _compute_tfidf_score(self, question, document):
        """Compute TF-IDF based relevance score (Jaccard similarity)."""
        q_terms = self._normalize_text(question)
        d_terms = self._normalize_text(document)
        
        if not q_terms or not d_terms:
            return 0.0
        
        overlap = len(q_terms & d_terms)
        union = len(q_terms | d_terms)
        
        return overlap / union if union > 0 else 0.0

    def _compute_term_coverage(self, question, document):
        """Compute what percentage of query terms appear in document."""
        q_terms = self._normalize_text(question)
        d_terms = self._normalize_text(document)
        
        if not q_terms:
            return 0.0
        
        covered = len(q_terms & d_terms)
        return covered / len(q_terms)

    def _compute_semantic_similarity(self, text1, text2):
        """Compute simple text similarity between two documents (word overlap)."""
        terms1 = self._normalize_text(text1)
        terms2 = self._normalize_text(text2)
        
        if not terms1 or not terms2:
            return 0.0
        
        overlap = len(terms1 & terms2)
        union = len(terms1 | terms2)
        
        return overlap / union if union > 0 else 0.0

    def _compute_cross_encoder_score(self, question, documents, batch_size=32):
        """Get cross-encoder scores."""
        doc_texts = [doc.get('text', '') for doc in documents]
        pairs = [[question, text] for text in doc_texts]
        
        scores = self.model.predict(pairs, batch_size=batch_size, convert_to_numpy=True)
        
        # Normalize to [0, 1] range
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores = np.ones_like(scores) * 0.5
            
        return scores

    def _safe_normalize(self, scores):
        """Safely normalize scores to [0, 1]."""
        if scores.max() > scores.min():
            return (scores - scores.min()) / (scores.max() - scores.min())
        return np.ones_like(scores) * 0.5

    def _rerank_basic(self, question, documents, top_k, batch_size):
        """Pure cross-encoder reranking."""
        if not documents:
            return []

        # Get cross-encoder scores
        scores = self._compute_cross_encoder_score(question, documents, batch_size)
        
        # Sort by score (descending) and get top-k indices
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        
        # Return top-k documents with scores
        ranked_docs = [
            {**documents[idx], '_rerank_score': float(scores[idx])} 
            for idx in ranked_indices
        ]
        
        return ranked_docs

    def _rerank_enhanced(self, question, documents, top_k, batch_size):
        """Multi-factor ensemble reranking."""
        if not documents:
            return []

        # Strategy 1: Cross-Encoder Score (50% weight)
        ce_scores = self._compute_cross_encoder_score(question, documents, batch_size)
        
        # Strategy 2: TF-IDF Score (30% weight)
        tfidf_scores = np.array([
            self._compute_tfidf_score(question, doc.get('text', '')) 
            for doc in documents
        ])
        
        # Strategy 3: Term Coverage (20% weight)
        coverage_scores = np.array([
            self._compute_term_coverage(question, doc.get('text', '')) 
            for doc in documents
        ])
        
        # Normalize all scores to [0, 1]
        ce_scores = self._safe_normalize(ce_scores)
        tfidf_scores = self._safe_normalize(tfidf_scores)
        coverage_scores = self._safe_normalize(coverage_scores)
        
        # Ensemble: weighted combination
        combined_scores = (
            0.50 * ce_scores +      # Cross-encoder dominates
            0.30 * tfidf_scores +   # Exact/partial matches
            0.20 * coverage_scores  # Query term coverage
        )
        
        # Sort by combined score (descending) and get top-k indices
        ranked_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        # Return top-k documents with their scores
        ranked_docs = [
            {**documents[idx], '_rerank_score': float(combined_scores[idx])} 
            for idx in ranked_indices
        ]
        
        return ranked_docs

    def _rerank_diversity(self, question, documents, top_k, batch_size):
        """Semantic relevance + diversity-aware reranking (MMR-inspired)."""
        if not documents:
            return []
        
        if len(documents) <= top_k:
            # If we have fewer docs than top_k, just score them
            ce_scores = self._compute_cross_encoder_score(question, documents, batch_size)
            ce_scores = self._safe_normalize(ce_scores)
            ranked_docs = [
                {**documents[idx], '_rerank_score': float(ce_scores[idx])} 
                for idx in np.argsort(ce_scores)[::-1]
            ]
            return ranked_docs
        
        # Get cross-encoder scores for all documents
        ce_scores = self._compute_cross_encoder_score(question, documents, batch_size)
        ce_scores = self._safe_normalize(ce_scores)
        
        # Iteratively select top-k documents with diversity penalty
        ranked_docs = []
        remaining_indices = list(range(len(documents)))
        
        for _ in range(top_k):
            if not remaining_indices:
                break
            
            best_idx_pos = None
            best_score = -1
            
            # For each remaining document, compute final score
            for pos, idx in enumerate(remaining_indices):
                doc_text = documents[idx].get('text', '')
                relevance_score = ce_scores[idx]
                
                # Compute diversity penalty: average similarity to already-selected docs
                diversity_penalty = 0.0
                if ranked_docs:
                    similarities = []
                    for selected_doc in ranked_docs:
                        sim = self._compute_semantic_similarity(
                            doc_text, 
                            selected_doc.get('text', '')
                        )
                        similarities.append(sim)
                    # Average similarity (lower is better for diversity)
                    diversity_penalty = np.mean(similarities)
                
                # Combined score: 60% relevance, 40% diversity (1 - redundancy)
                final_score = 0.60 * relevance_score + 0.40 * (1 - diversity_penalty)
                
                if final_score > best_score:
                    best_score = final_score
                    best_idx_pos = pos
            
            # Select best document
            if best_idx_pos is not None:
                best_idx = remaining_indices.pop(best_idx_pos)
                ranked_docs.append({
                    **documents[best_idx],
                    '_rerank_score': float(0.60 * ce_scores[best_idx])
                })
        
        return ranked_docs

    def rerank(self, question, documents, top_k=15, batch_size=32):
        """
        Rerank documents using the chosen strategy.

        Args:
            question: Question string
            documents: List of document dicts with 'text' field
            top_k: Number of top documents to return
            batch_size: Batch size for scoring

        Returns:
            List of top-k documents sorted by relevance (highest first)
        """
        if self.strategy == "basic":
            return self._rerank_basic(question, documents, top_k, batch_size)
        elif self.strategy == "enhanced":
            return self._rerank_enhanced(question, documents, top_k, batch_size)
        else:  # diversity
            return self._rerank_diversity(question, documents, top_k, batch_size)


def load_reranker(strategy="enhanced", use_fp16=True):
    """
    Load reranker with specified strategy.

    Args:
        strategy: "basic" (cross-encoder only) or "enhanced" (multi-factor)
        use_fp16: Use FP16 precision

    Returns:
        Reranker instance
    """
    return Reranker(strategy=strategy, use_fp16=use_fp16)


def rerank_documents(question, documents, reranker, top_k=15):
    """
    Rerank documents for a question using reranker instance.

    Args:
        question: Question string
        documents: List of document dicts with 'text' field
        reranker: Reranker instance
        top_k: Number of top documents to return

    Returns:
        List of top-k documents sorted by relevance
    """
    return reranker.rerank(question, documents, top_k=top_k)
