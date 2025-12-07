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

# 5. Train the model (REQUIRED - model not included in repo due to size)
python3 -m src.fine_tuning.fine_tune
# Training takes 20-40 minutes on CPU
# Creates: finetuned-accessibility-bert/final_model/
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

⚠️ **IMPORTANT**: The trained model is NOT included in this repo due to GitHub's 100MB file size limit. You must train it locally before using the pipeline.

**Train Model (REQUIRED on first setup):**
```bash
python3 -m src.fine_tuning.fine_tune
```
- Takes 20-40 minutes on CPU
- Creates `finetuned-accessibility-bert/final_model/` directory (~500MB)
- Only needs to be done once

**Evaluate Model:**
```bash
python3 -m src.fine_tuning.evaluate
```

**Dataset:** 526 training reviews, 132 test reviews (mall reviews from Google Maps)

---

## 🔌 Backend Integration Guide (For Backend Dev)

### Analyzing Multiple Reviews

```
STEP 1: Initialize pipeline (only once at startup)
    pipeline = AccessibilityPipeline()

STEP 2: Get reviews from database
    reviews = fetch_reviews_from_database()

STEP 3: Process each review
    FOR each review in reviews:
        TRY:
            result = pipeline.analyze_review(review.text)
            
            Store result:
                - is_accessibility_related
                - keywords
                - sentiment ("Positive" | "Mixed" | "Negative")
                - sentiment_score (0.0 to 1.0)
        
        CATCH SystemExit:
            // Review skipped (not accessibility-related)
            Store NULL values:
                - is_accessibility_related = False
                - keywords = []
                - sentiment = NULL
                - sentiment_score = 0.0
        
        CATCH Exception:
            // Error occurred
            Store NULL values for all fields

STEP 4: Save results to database
```

### Important Notes for Backend Dev

1. **Initialize Once**: Create pipeline instance at app startup, NOT per request
2. **Handle Skipped Reviews**: Store NULL/0 values when review is skipped (not accessibility-related)
3. **Performance**: First analysis takes ~1 minute (model loading), then ~200ms per review
4. **Error Handling**: Wrap in try-except to handle empty text, encoding issues
5. **Batch Processing**: Process reviews in batches for efficiency

### Backend Notes

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
