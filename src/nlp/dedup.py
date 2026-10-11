from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class NoveltyScorer:
    """Novelty = 1 - max cosine similarity to recent items."""

    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self.history: list[str] = []
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)

    def novelty(self, text: str) -> float:
        if not self.history:
            self.history.append(text)
            return 1.0
        recent = self.history[-self.lookback:]
        corpus = recent + [text]
        try:
            X = self.vec.fit_transform(corpus)
            sims = cosine_similarity(X[-1], X[:-1]).flatten()
            max_sim = float(sims.max()) if len(sims) else 0.0
        except ValueError:
            max_sim = 0.0
        self.history.append(text)
        return max(0.0, min(1.0, 1.0 - max_sim))
