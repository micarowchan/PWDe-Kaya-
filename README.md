# PWDeKaya Sentiment Analysis Module

Sentiment analysis system for analyzing accessibility-related reviews of establishments using fine-tuned RoBERTa model.

---

## 📊 Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 87% |
| **F1-Score** | 83% |
| **Model** | RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest) |

---

## ✅ TODO: Backend Integration

**Next Steps:**
1. Connect this module to Firebase backend
2. Ensure reviews per establishment are stored in the database
3. Calculate **median sentiment score** per establishment from all reviews
4. Store aggregated sentiment alongside establishment data

**Implementation Notes:**
- Use `AccessibilityPipeline` to analyze each review
- Store individual results: `sentiment`, `sentiment_score`, `keywords`, `is_accessibility_related`
- Compute median of `sentiment_score` values for overall establishment rating
- Update establishment document with aggregated sentiment

---

## 📁 Project Structure

```
sentiment_module/
├── README.md                          # Documentation
├── requirements.txt                   # Python dependencies
├── evaluation_results.json            # Model performance metrics
├── data/
│   ├── train_reviews.csv             # Training dataset (526 reviews)
│   └── test_reviews.csv              # Test dataset (132 reviews)
├── src/
│   ├── __init__.py
│   ├── accessibility_pipeline.py     # Main pipeline orchestrator
│   ├── preprocessing/
│   │   ├── text_preprocessor.py      # Text cleaning and tokenization
│   │   ├── keyword_extractor.py      # Semantic keyword extraction
│   │   ├── keyword_manager.py        # Keyword database manager
│   │   ├── feature_extractor.py      # N-gram generation
│   │   └── __init__.py
│   ├── model/
│   │   ├── sentiment_model.py        # RoBERTa sentiment classifier
│   │   └── __init__.py
│   ├── fine_tuning/
│   │   ├── fine_tune.py              # Model training with class weights
│   │   ├── evaluate.py               # Evaluation and JSON export
│   │   └── __init__.py
│   └── utils/
│       └── accessibility_keywords.json  # Accessibility keyword database
└── finetuned-accessibility-bert/
    └── final_model/                   # Trained RoBERTa model
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        └── vocab.txt
```

---

## 🚀 Quick Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # WSL/Linux/Mac
# OR: venv\Scripts\activate  # Windows

# 2. Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Install dependencies
pip install transformers datasets nltk sentence-transformers scikit-learn pandas

# 4. Download NLTK data
python3 -c "import nltk; nltk.download(['wordnet', 'omw-1.4', 'stopwords', 'averaged_perceptron_tagger'])"
```

---

## 💻 Usage

```python
from src.accessibility_pipeline import AccessibilityPipeline

# Initialize once (reuse for all reviews)
pipeline = AccessibilityPipeline()

# Analyze a review
result = pipeline.analyze_review("The mall has wheelchair ramps and accessible parking")

# Returns:
# {
#     'is_accessibility_related': True,
#     'keywords': ['wheelchair', 'ramps', 'accessible', 'parking'],
#     'sentiment': 'Positive',
#     'confidence': 0.92
# }
```

---

## 🎓 Training & Evaluation

**Train Model:**
```bash
python3 -m src.fine_tuning.fine_tune
```

**Evaluate Model:**
```bash
python3 -m src.fine_tuning.evaluate
```

**Dataset:** 526 training reviews, 132 test reviews (mall reviews from Google Maps)

---

## 🔌 Backend Integration Guide (For Backend Dev)

### Analyzing Multiple Reviews

```python
from src.accessibility_pipeline import AccessibilityPipeline

# STEP 1: Initialize pipeline ONCE (takes ~1 minute on first load)
# Reuse this instance for ALL reviews
pipeline = AccessibilityPipeline()

# STEP 2: Get all reviews from your database
# Example structure:
reviews = [
    {"id": "review_123", "text": "Great wheelchair access", "establishment_id": "mall_1"},
    {"id": "review_456", "text": "Food was delicious", "establishment_id": "mall_1"},
    # ... more reviews
]

# STEP 3: Process each review
results = []

for review in reviews:
    try:
        # Analyze the review
        result = pipeline.analyze_review(review["text"])
        
        # Store the result with review ID
        results.append({
            "review_id": review["id"],
            "establishment_id": review["establishment_id"],
            "is_accessibility_related": result["is_accessibility_related"],
            "keywords": result["keywords"],
            "sentiment": result["sentiment"],  # "Positive" | "Mixed" | "Negative"
            "sentiment_score": result["sentiment_score"]  # 0.0 to 1.0
        })
        
    except SystemExit:
        # Review was skipped (not accessibility-related)
        # IMPORTANT: Store NULL/zero values for skipped reviews
        results.append({
            "review_id": review["id"],
            "establishment_id": review["establishment_id"],
            "is_accessibility_related": False,
            "keywords": [],
            "sentiment": None,  # NULL in database
            "sentiment_score": 0.0  # Zero for numeric score
        })
        
    except Exception as e:
        # Handle other errors (e.g., empty text, encoding issues)
        print(f"Error processing review {review['id']}: {e}")
        results.append({
            "review_id": review["id"],
            "establishment_id": review["establishment_id"],
            "is_accessibility_related": None,  # NULL - error state
            "keywords": [],
            "sentiment": None,
            "sentiment_score": 0.0
        })

# STEP 4: Batch insert results into database
# Insert all results into your reviews table or sentiment_results table
```

### Database Schema Recommendation

```sql
-- Add these columns to your reviews table
ALTER TABLE reviews ADD COLUMN is_accessibility_related BOOLEAN;
ALTER TABLE reviews ADD COLUMN keywords TEXT[];  -- Array of strings
ALTER TABLE reviews ADD COLUMN sentiment VARCHAR(10);  -- "Positive", "Mixed", "Negative", or NULL
ALTER TABLE reviews ADD COLUMN sentiment_score DECIMAL(3,2);  -- 0.00 to 1.00

-- Or create separate table
CREATE TABLE review_sentiments (
    review_id VARCHAR PRIMARY KEY,
    establishment_id VARCHAR,
    is_accessibility_related BOOLEAN,
    keywords TEXT[],
    sentiment VARCHAR(10),
    sentiment_score DECIMAL(3,2),
    analyzed_at TIMESTAMP DEFAULT NOW()
);
```

### Calculating Establishment Median Sentiment

```python
def calculate_establishment_sentiment(establishment_id):
    """
    Calculate median sentiment for an establishment from all its reviews
    """
    # Get all accessibility-related reviews for this establishment
    # Filter: is_accessibility_related = True AND sentiment IS NOT NULL
    reviews = get_reviews_from_db(
        establishment_id=establishment_id,
        is_accessibility_related=True
    )
    
    # Extract sentiment scores (skip NULL values)
    sentiment_scores = [
        r["sentiment_score"] 
        for r in reviews 
        if r["sentiment_score"] is not None and r["sentiment_score"] > 0
    ]
    
    if len(sentiment_scores) == 0:
        return None  # No valid reviews
    
    # Calculate median
    from statistics import median
    median_score = median(sentiment_scores)
    
    # Update establishment record
    update_establishment(establishment_id, {
        "accessibility_sentiment_median": median_score,
        "accessibility_review_count": len(sentiment_scores)
    })
    
    return median_score
```

### Important Notes for Backend 

1. **Initialize Once**: Create pipeline instance at app startup, NOT per request
2. **Handle Skipped Reviews**: Store NULL/0 values when review is skipped (not accessibility-related)
3. **Median Calculation**: Only use reviews where `is_accessibility_related = True` and `sentiment_score > 0`
4. **Performance**: First analysis takes ~5 min (model loading), then ~200ms per review
5. **Error Handling**: Wrap in try-except to handle empty text, encoding issues
6. **Batch Processing**: Process reviews in batches, insert results in bulk for efficiency

### Backend Notes

- After makuha yung sentiment sa gabos, i kuha na lang yung median kang gabos na review sentiment tapos yung average ng confidence score, per establishment, tapos i- store.

- Sa paglagay ng sa evaluation. Nasa evaluation_results.json bale ifefetch na lang dito sa json file yung evaluation results since same man sa lahat na establishments.

- After makuha yung sentiment sa gabos, i kuha na lang yung median kang gabos na review sentiment tapos yung average ng confidence score, per establishment, tapos i- store.

- Sa paglagay ng sa evaluation. Nasa evaluation_results.json bale ifefetch na lang dito sa json file yung evaluation results since same man sa lahat na establishments.

---

## ⚠️ Important Notes

- **CPU-only**: No GPU required, uses PyTorch CPU version
- **Model**: Fine-tuned RoBERTa saved in `finetuned-accessibility-bert/final_model/` (~500MB)
- **Training time**: 20-40 minutes on CPU
- **Initialize once**: Reuse pipeline instance across all requests for best performance

---

## 🛠️ Common Issues

**"No module named 'src'"**
```bash
cd sentiment_module
python3 -m src.fine_tuning.fine_tune
```

**"NLTK resource not found"**
```bash
python3 -c "import nltk; nltk.download(['wordnet', 'omw-1.4', 'stopwords', 'averaged_perceptron_tagger'])"
```

**Training too slow**
- Expected: 20-40 minutes on CPU
- Reduce epochs from 5 to 3 in `fine_tune.py`

---

**Status**: ✅ Production-ready | 87% accuracy | December 2025
