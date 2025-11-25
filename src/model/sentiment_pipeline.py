import statistics

class SentimentPipeline:
    def __init__(self, sentiment_model):
        self.model = sentiment_model

    def analyze_reviews(self, reviews: list[str]) -> dict:
        scores = []
        confidences = []

        for review in reviews:
            result = self.model.analyze_sentiment(review)
            score = result["score"]
            confidence = result["confidence"]

            scores.append(score)
            confidences.append(confidence)

        median_score = statistics.median(scores) if scores else None
        avg_confidence = sum(confidences) / len(confidences) if confidences else None

        overall = self.classify_overall(median_score)

        return {
            "median_score": round(median_score, 1) if median_score else None,
            "average_confidence": round(avg_confidence, 2) if avg_confidence else None,
            "overall_sentiment": overall,
            "review_count": len(reviews)
        }

    def classify_overall(self, score: float) -> str:
        if score is None:
            return "no_reviews"

        if score >= 3.6:
            return "positive"
        elif score <= 2.4:
            return "negative"
        else:
            return "mixed"
