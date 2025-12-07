from sentence_transformers import SentenceTransformer, util
import torch

class KeywordExtractor:
    def __init__(self, keyword_manager, sbert_model="all-mpnet-base-v2", debug=False):
        self.manager = keyword_manager
        self.model = SentenceTransformer(sbert_model)
        self.keyword_embeddings = self.model.encode(self.manager.keywords, convert_to_tensor=True)
        self.debug = debug

    def extract_keywords(self, ngrams: list, threshold_match=0.85, threshold_expand=0.9, expand_keywords=True):
        """
        Extract accessibility keywords from ngrams using semantic similarity
        
        Args:
            ngrams: List of n-gram strings to match against keywords
            threshold_match: Minimum similarity score to match existing keywords
            threshold_expand: Minimum similarity score to add new keywords
            expand_keywords: Whether to dynamically add new similar keywords
        
        Returns:
            List of matched accessibility keywords
        """
        if not ngrams:
            return []

        # Encode ngrams and calculate cosine similarity with keyword embeddings
        ngram_embeddings = self.model.encode(ngrams, convert_to_tensor=True)
        cosine_scores = util.cos_sim(ngram_embeddings, self.keyword_embeddings)

        extracted_keywords = set()
        new_keywords = []

        # Iterate through ngrams and their cosine scores
        for i, ngram in enumerate(ngrams):
            for j, keyword in enumerate(self.manager.keywords):
                score = float(cosine_scores[i][j])
                if score >= threshold_match:
                    extracted_keywords.add(keyword)
                    if self.debug:
                        print(f"Matched: '{ngram}' -> '{keyword}' (score: {score:.3f})")

            # Check for potential new keywords to expand the list
            if expand_keywords:
                max_score = float(torch.max(cosine_scores[i]).item())
                if max_score >= threshold_expand and ngram not in self.manager.keywords:
                    new_keywords.append(ngram)
                    if self.debug:
                        print(f"New keyword discovered: '{ngram}' (score: {max_score:.3f})")

        # Update keyword manager with new keywords
        if new_keywords:
            self.manager.add_keywords(new_keywords, save=False)
            self.keyword_embeddings = self.model.encode(self.manager.keywords, convert_to_tensor=True)

        return list(extracted_keywords)
    
    def is_relevant_review(self, ngrams: list, threshold_match: float = 0.85) -> bool:
        """
        Check if review is accessibility-related based on keyword matching
        
        Args:
            ngrams: List of n-gram strings from the review
            threshold_match: Minimum similarity score to consider relevant
        
        Returns:
            True if accessibility keywords are found, False otherwise
        """
        return bool(self.extract_keywords(ngrams, threshold_match=threshold_match, expand_keywords=False))
