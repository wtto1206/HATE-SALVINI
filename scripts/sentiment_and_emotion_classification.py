"""SCRIPT TO COMPUTE EMOTION AND SENTIMENT CLASSIFIFICATION ON THE IMMIGRATION_RELATED TWEETS AND CREATE PIE CHARTS"""

import pandas as pd
import matplotlib.pyplot as plt
from feel_it import EmotionClassifier, SentimentClassifier

# Load relevant tweets
relevant_tweets = pd.read_csv('all_topics.csv')  # Adjust path as needed

# Initialize emotion and sentiment classifiers
emotion_classifier = EmotionClassifier()
sentiment_classifier = SentimentClassifier()

# Dictionary to store sentiment and emotion predictions
sentiment_emotion_dict = {}

# Predict sentiment and emotion for each tweet
for tweet in relevant_tweets['Doc']:
    sentiment = sentiment_classifier.predict([tweet])[0]
    emotion = emotion_classifier.predict([tweet])[0]
    sentiment_emotion_dict[tweet] = {"sentiment": sentiment, "emotion": emotion}

# Create DataFrame from sentiment_emotion_dict
df = pd.DataFrame(sentiment_emotion_dict).T

# Calculate percentage of tweets with each emotion
emotion_counts = df["emotion"].value_counts(normalize=True) * 100

# Calculate percentage of tweets with each sentiment
sentiment_counts = df["sentiment"].value_counts(normalize=True) * 100

# Define colors for emotions (adjust as needed)
colors_emotion = ['#EAD2AC', '#B2CCB2', '#6CA9A5', '#545E75']

# Plot pie charts for sentiment and emotion distributions
plt.figure(figsize=(16, 8))

# Plot pie chart for sentiment distribution
plt.subplot(1, 2, 1)
wedges_sentiment, _, autotexts_sentiment = plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=colors_emotion)
plt.title('Sentiment Distribution')
plt.axis('equal')

# Add legend for sentiment
plt.legend(wedges_sentiment, sentiment_counts.index, title="Sentiments", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

# Plot pie chart for emotion distribution
plt.subplot(1, 2, 2)
wedges_emotion, _, autotexts_emotion = plt.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=140, colors=colors_emotion)
plt.title('Emotion Distribution')
plt.axis('equal')

# Add legend for emotion
plt.legend(wedges_emotion, emotion_counts.index, title="Emotions", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the combined plot as an image
plt.savefig('sentiment_and_emotion_pie_charts.png', dpi=300, bbox_inches='tight')

# Display the combined plot
plt.show()
