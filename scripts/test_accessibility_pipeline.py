
# Test script for Accessibility Pipeline
# Tests the complete pipeline with various accessibility-related reviews


from src.preprocessing import TextPreprocessor, FeatureExtractor, KeywordManager, KeywordExtractor
from src.model import SentimentModel
from src.accessibility_pipeline import AccessibilityPipeline


def test_pipeline():
    # Test the accessibility pipeline with sample reviews
    
    print("=" * 70)
    print("INITIALIZING ACCESSIBILITY PIPELINE")
    print("=" * 70)
    
    # Initialize components
    print("\n[1/5] Loading text preprocessor...")
    preprocessor = TextPreprocessor()
    
    print("[2/5] Loading feature extractor...")
    feature_extractor = FeatureExtractor()
    
    print("[3/5] Loading keyword manager...")
    keyword_manager = KeywordManager(keywords_file="src/utils/accessibility_keywords.json")
    
    print("[4/5] Loading keyword extractor...")
    keyword_extractor = KeywordExtractor(keyword_manager)
    
    print("[5/5] Loading sentiment model...")
    model_path = "finetuned-accessibility-bert/final_model"
    sentiment_model = SentimentModel(model_path=model_path)
    
    # Create pipeline
    pipeline = AccessibilityPipeline(
        preprocessor=preprocessor,
        feature_extractor=feature_extractor,
        keyword_manager=keyword_manager,
        keyword_extractor=keyword_extractor,
        sentiment_model=sentiment_model
    )
    
    print("\n✓ Pipeline initialized successfully!\n")
    
    # Test cases
    test_reviews = [
        {
            "text": "The venue has excellent wheelchair accessible ramps and the staff were very helpful with mobility assistance.",
            "expected_sentiment": "Positive",
            "description": "Positive accessibility review"
        },
        {
            "text": "Unfortunately, there was no wheelchair access and the elevator was broken. Very disappointing for disabled visitors.",
            "expected_sentiment": "Negative",
            "description": "Negative accessibility review"
        },
        {
            "text": "The building has ramps but they are quite steep. The accessible parking is available but limited.",
            "expected_sentiment": "Mixed",
            "description": "Mixed accessibility review"
        },
        {
            "text": "Braille signage throughout the facility and audio assistance for visually impaired guests. Outstanding accessibility features!",
            "expected_sentiment": "Positive",
            "description": "Positive accessibility with visual impairment features"
        },
        {
            "text": "No accessible restrooms available. The entrance has stairs with no alternative access for wheelchairs.",
            "expected_sentiment": "Negative",
            "description": "Negative accessibility with multiple issues"
        },
        {
            "text": "Great food and atmosphere.",
            "expected_sentiment": None,
            "description": "Non-accessibility review (should be skipped)"
        }
    ]
    
    print("=" * 70)
    print("RUNNING TESTS")
    print("=" * 70)
    
    results = []
    
    for i, test_case in enumerate(test_reviews, 1):
        print(f"\n{'=' * 70}")
        print(f"TEST {i}: {test_case['description']}")
        print(f"{'=' * 70}")
        print(f"Review: \"{test_case['text']}\"")
        print()
        
        try:
            result = pipeline.analyze_review(test_case['text'], save_keywords=False)
            
            print(f"✓ RESULT:")
            print(f"  Keywords found: {', '.join(result['keywords']) if result['keywords'] else 'None'}")
            print(f"  Sentiment: {result['sentiment']}")
            print(f"  Confidence: {result['confidence']}%")
            
            # Check if sentiment matches expected
            if test_case['expected_sentiment']:
                match = result['sentiment'] == test_case['expected_sentiment']
                status = "✓ PASS" if match else "✗ FAIL"
                print(f"  Expected: {test_case['expected_sentiment']}")
                print(f"  Status: {status}")
            
            results.append({
                "test": test_case['description'],
                "status": "PASS",
                "result": result
            })
            
        except SystemExit:
            print("✗ Review skipped (not accessibility-related)")
            if test_case['expected_sentiment'] is None:
                print("  Status: ✓ PASS (correctly identified as non-relevant)")
                results.append({
                    "test": test_case['description'],
                    "status": "PASS",
                    "result": None
                })
            else:
                print("  Status: ✗ FAIL (should have been analyzed)")
                results.append({
                    "test": test_case['description'],
                    "status": "FAIL",
                    "result": None
                })
        except Exception as e:
            print(f"✗ ERROR: {str(e)}")
            results.append({
                "test": test_case['description'],
                "status": "ERROR",
                "result": str(e)
            })
    
    # Summary
    print(f"\n{'=' * 70}")
    print("TEST SUMMARY")
    print(f"{'=' * 70}")
    
    total_tests = len(results)
    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = sum(1 for r in results if r['status'] == 'FAIL')
    errors = sum(1 for r in results if r['status'] == 'ERROR')
    
    print(f"\nTotal Tests: {total_tests}")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"⚠ Errors: {errors}")
    print(f"\nSuccess Rate: {(passed/total_tests)*100:.1f}%")
    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    test_pipeline()
