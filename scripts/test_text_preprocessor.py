from src.preprocessing.text_preprocessor import TextPreprocessor

text_preprocessor = TextPreprocessor(keywords_file="src/utils/accessibility_keywords.json", debug=True)

review_samples = [
    "It's okay. They're PWD-friendly. They do have a person or staff at the lobby but not as approachable as they should be. Went here a couple of times to view exhibits and when I try to ask, I find they are not that friendly but had a nice visit anyway.",
    "They have a few parking slots that are shared with the nearby establishments. Items are sorted neatly according to purpose, so it's easy to find exactly where to get what you need. They accept credit card payment. They have a self service GoTyme kiosk, as well as a counter dedicated for GoTyme deposit and withdrawal.",
]

for review in review_samples:
    print("Review:", review)
    keywords = text_preprocessor.extract_accessibility_keywords(review, expand_keywords=True, threshold=0.75)
    print("Accessibility Keywords:", keywords)
    print("\n")

