"""
Filter reviews to identify which ones are relevant to accessibility
Uses the KeywordExtractor to check if reviews contain accessibility-related keywords
"""

import pandas as pd
import sys
sys.path.append('.')

from src.preprocessing import TextPreprocessor, FeatureExtractor, KeywordManager, KeywordExtractor

def filter_accessibility_reviews(input_csv, output_relevant_csv, output_non_relevant_csv):
    """
    Filter reviews based on accessibility relevance
    
    Args:
        input_csv: Path to input CSV file with reviews
        output_relevant_csv: Path to save accessibility-relevant reviews
        output_non_relevant_csv: Path to save non-relevant reviews
    """
    
    print("=" * 70)
    print("FILTERING ACCESSIBILITY REVIEWS")
    print("=" * 70)
    
    # Initialize components
    print("\n[1/4] Loading preprocessor and feature extractor...")
    preprocessor = TextPreprocessor()
    feature_extractor = FeatureExtractor()
    
    print("[2/4] Loading keyword manager...")
    keyword_manager = KeywordManager(keywords_file="src/utils/accessibility_keywords.json")
    print(f"      Loaded {len(keyword_manager.keywords)} accessibility keywords")
    
    print("[3/4] Initializing keyword extractor with SBERT...")
    keyword_extractor = KeywordExtractor(keyword_manager)
    
    print("[4/4] Loading reviews dataset...")
    df = pd.read_csv(input_csv)
    print(f"      Total reviews: {len(df)}")
    
    print("\n" + "=" * 70)
    print("ANALYZING REVIEWS FOR ACCESSIBILITY RELEVANCE")
    print("=" * 70)
    
    relevant_reviews = []
    non_relevant_reviews = []
    
    for idx, row in df.iterrows():
        review_text = row['review_text']
        label = row['label']
        
        # Preprocess text
        cleaned_text = preprocessor.lowercase(review_text)
        cleaned_text = preprocessor.remove_url(cleaned_text)
        cleaned_text = preprocessor.remove_non_alpha(cleaned_text)
        
        # Tokenize and preprocess
        tokens = cleaned_text.split()
        tokens = preprocessor.remove_stopwords(tokens)
        tokens = preprocessor.remove_articles(tokens)
        
        # Lemmatize and filter
        lemmatized_tokens = preprocessor.lemmatize(tokens)
        filtered_tokens = preprocessor.filter_pos(lemmatized_tokens)
        
        # Generate n-grams
        ngrams = feature_extractor.generate_ngrams(filtered_tokens)
        
        # Check if review is accessibility-related
        is_relevant = keyword_extractor.is_relevant_review(ngrams, threshold_match=0.85)
        
        if is_relevant:
            # Extract keywords to see what matched
            keywords = keyword_extractor.extract_keywords(ngrams, threshold_match=0.85, expand_keywords=False)
            relevant_reviews.append({
                'review_text': review_text,
                'label': label,
                'keywords': ', '.join(keywords) if keywords else ''
            })
            status = "✓ RELEVANT"
            kw_info = f" ({len(keywords)} keywords)" if keywords else ""
        else:
            non_relevant_reviews.append({
                'review_text': review_text,
                'label': label
            })
            status = "✗ NOT RELEVANT"
            kw_info = ""
        
        # Print progress every 50 reviews
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(df)} reviews...")
    
    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    relevant_df = pd.DataFrame(relevant_reviews)
    non_relevant_df = pd.DataFrame(non_relevant_reviews)
    
    relevant_df.to_csv(output_relevant_csv, index=False)
    non_relevant_df.to_csv(output_non_relevant_csv, index=False)
    
    print(f"\n✓ Saved {len(relevant_reviews)} accessibility-relevant reviews to:")
    print(f"  {output_relevant_csv}")
    print(f"\n✓ Saved {len(non_relevant_reviews)} non-relevant reviews to:")
    print(f"  {output_non_relevant_csv}")
    
    # Statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total = len(df)
    relevant_count = len(relevant_reviews)
    non_relevant_count = len(non_relevant_reviews)
    
    print(f"\nTotal reviews analyzed: {total}")
    print(f"✓ Accessibility-relevant: {relevant_count} ({relevant_count/total*100:.1f}%)")
    print(f"✗ Not relevant: {non_relevant_count} ({non_relevant_count/total*100:.1f}%)")
    
    # Label distribution in relevant reviews
    if relevant_count > 0:
        relevant_labels = relevant_df['label'].value_counts().sort_index()
        print(f"\nLabel distribution in accessibility-relevant reviews:")
        print(f"  0 (Negative): {relevant_labels.get(0, 0)}")
        print(f"  1 (Mixed): {relevant_labels.get(1, 0)}")
        print(f"  2 (Positive): {relevant_labels.get(2, 0)}")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    input_csv = "data/reviews.csv"
    output_relevant = "data/reviews_accessibility.csv"
    output_non_relevant = "data/reviews_non_accessibility.csv"
    
    filter_accessibility_reviews(input_csv, output_relevant, output_non_relevant)
