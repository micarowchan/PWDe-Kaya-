class FeatureExtractor:
    def __init__(self, max_n=3):
        self.max_n = max_n

    def generate_ngrams(self, tokens: list) -> list:
        ngrams = []
        for n in range(1, self.max_n + 1):
            for i in range(len(tokens)-n+1):
                ngram_tokens = tokens[i:i+n]
                ngrams.append(" ".join(ngram_tokens))
        return ngrams
