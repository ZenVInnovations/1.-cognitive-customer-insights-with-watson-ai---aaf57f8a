# Ensure the transformers library is installed in your environment before running this script.
# You can install it using the following command in your terminal:
# pip install transformers
import pandas as pd
from transformers import pipeline

# Load sentiment analysis and NER pipelines
sentiment_analysis = pipeline("sentiment-analysis")
ner_model = pipeline("ner", aggregation_strategy="simple")

# Example customer feedback
customer_feedback = [
    "I love the product! It's exactly what I needed.",
    "I'm very disappointed with the service. It was too slow and frustrating.",
    "The quality is great, but the price is too high.",
    "I bought an iPhone at the Apple Store in San Francisco.",
    "Amazon’s delivery was fast but the package was damaged."
]

# Prepare a list to store results
data = []

# Analyze each feedback for sentiment and entities
for feedback in customer_feedback:
    sentiment_result = sentiment_analysis(feedback)[0]
    sentiment = sentiment_result['label']
    confidence = sentiment_result['score']

    entities = ner_model(feedback)
    # Extract key entity words (e.g., orgs, products, locations)
    extracted_entities = [entity['word'] for entity in entities if entity['entity_group'] in ['ORG', 'PRODUCT', 'LOC']]

    # Append the results to the data list
    data.append([feedback, sentiment, confidence, extracted_entities])

    # Print feedback analysis
    print(f"Feedback: {feedback}")
    print(f"Sentiment: {sentiment} (confidence: {confidence:.2f})")
    print(f"Entities: {extracted_entities}")
    print("-" * 80)

# Create a DataFrame for visualization
df = pd.DataFrame(data, columns=['Feedback', 'Sentiment', 'Confidence', 'Entities'])
print(df)
