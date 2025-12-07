
# Test script for Sentiment Model
# Tests if the sentiment model can classify reviews correctly


from src.model.sentiment_model import SentimentModel

def test_sentiment_model():
    print("=" * 70)
    print("TESTING SENTIMENT MODEL")
    print("=" * 70)
    
    # Initialize model
    print("\n[1/2] Loading sentiment model...")
    try:
        model = SentimentModel()
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Test cases
    test_reviews = [
        {
            "text": "Excellent service and very clean facilities!",
            "expected": "Positive",
            "description": "Clearly positive review"
        },
        {
            "text": "Terrible experience, staff were rude and unhelpful.",
            "expected": "Negative",
            "description": "Clearly negative review"
        },
        {
            "text": "Good food but service was slow and parking was difficult.",
            "expected": "Mixed",
            "description": "Mixed sentiment review"
        },
        {
            "text": "The wheelchair ramps are well-maintained and accessible.",
            "expected": "Positive",
            "description": "Positive accessibility review"
        },
        {
            "text": "No accessibility features at all, very disappointing.",
            "expected": "Negative",
            "description": "Negative accessibility review"
        },
    ]
    
    print(f"\n[2/2] Running {len(test_reviews)} test cases...\n")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_reviews, 1):
        text = test["text"]
        expected = test["expected"]
        description = test["description"]
        
        print(f"\nTest {i}/{len(test_reviews)}: {description}")
        print(f"Review: \"{text}\"")
        print(f"Expected: {expected}")
        
        try:
            result = model.analyze_sentiment(text)
            predicted = result['sentiment']
            confidence = result['confidence']
            
            print(f"Predicted: {predicted} (confidence: {confidence:.2f}%)")
            
            if predicted == expected:
                print("✓ PASS")
                passed += 1
            else:
                print("✗ FAIL")
                failed += 1
                
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
        
        print("-" * 70)
    
    # Summary
    print(f"\n{'=' * 70}")
    print("TEST SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total: {len(test_reviews)}")
    print(f"Passed: {passed} ✓")
    print(f"Failed: {failed} ✗")
    print(f"Success Rate: {(passed/len(test_reviews)*100):.1f}%")
    print(f"{'=' * 70}\n")
    
    return failed == 0

if __name__ == "__main__":
    success = test_sentiment_model()
    exit(0 if success else 1)
