""" CODE TO COMPUTE FREQUENCY DISTRIBUTION OVER THE DATA AND CHECK N-GRAMS"""

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import spacy
from stop_words import get_stop_words

# Load cleaned data
df = pd.read_csv('cleaned_data.csv')

# Italian stop words
italian_stop_words = get_stop_words('italian')

# Load Italian spaCy model
nlp = spacy.load("it_core_news_sm")

def tokenize_tweets(df):
    """
    Tokenize tweets using spaCy and remove stop words and punctuation.
    
    Args:
    df (pd.DataFrame): DataFrame containing tweets.
    
    Returns:
    list: List of tokenized tweets.
    """
    tokenized_tweets = []
    for tweet in df['tweet']:
        doc = nlp(tweet)
        tokenized_tweet = [token.text.lower() for token in doc if token.is_alpha and token.text.lower() not in italian_stop_words and not token.is_punct]
        tokenized_tweets.append(tokenized_tweet)
    return tokenized_tweets

def plot_most_common(tokens_freq, title, filename):
    """
    Plot and save bar chart for most common tokens, bigrams, or trigrams.
    
    Args:
    tokens_freq (list): List of tuples containing tokens and their frequencies.
    title (str): Title of the plot.
    filename (str): Filename to save the plot.
    """
    tokens_freq.reverse()  # Reverse order for plotting
    tokens = [token for token, _ in tokens_freq]
    frequencies = [freq for _, freq in tokens_freq]
    
    plt.figure(figsize=(16, 8))
    plt.barh(tokens, frequencies, color='#545E75')
    plt.title(title, fontsize=15, weight='bold')
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Tokens', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.legend(['Count'])
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')

def main():
    # Tokenize all tweets
    tokenized_tweets = tokenize_tweets(df)
    
    # Flatten tokenized tweets into a single list
    all_tokens = [token for tweet in tokenized_tweets for token in tweet]
    
    # Calculate frequency distribution for unigrams
    FreqDict = Counter(all_tokens)
    most_common_unigrams = FreqDict.most_common(50)
    plot_most_common(most_common_unigrams, '50 Most Common Unigrams in the Tweets (General)', '50_most_common_unigrams_general.png')
    
    # Calculate frequency distribution for bigrams
    bigrams = list(zip(all_tokens, all_tokens[1:]))
    bigram_freq = Counter(bigrams)
    most_common_bigrams = bigram_freq.most_common(50)
    plot_most_common(most_common_bigrams, '50 Most Common Bigrams in the Tweets (General)', '50_most_common_bigrams_general.png')
    
    # Calculate frequency distribution for trigrams
    trigrams = list(zip(all_tokens, all_tokens[1:], all_tokens[2:]))
    trigram_freq = Counter(trigrams)
    most_common_trigrams = trigram_freq.most_common(50)
    plot_most_common(most_common_trigrams, '50 Most Common Trigrams in the Tweets (General)', '50_most_common_trigrams_general.png')
    
    # Relevant tweets analysis
    relevant_tweets = pd.read_csv('all_topics.csv')  # Adjust path as needed
    
    # Tokenize relevant tweets
    tokenized_relevant_tweets = tokenize_tweets(relevant_tweets)
    
    # Flatten tokenized relevant tweets into a single list
    all_tokens_relevant = [token for tweet in tokenized_relevant_tweets for token in tweet]
    
    # Calculate frequency distribution for unigrams in relevant tweets
    FreqDict_relevant = Counter(all_tokens_relevant)
    most_common_unigrams_relevant = FreqDict_relevant.most_common(50)
    plot_most_common(most_common_unigrams_relevant, '50 Most Common Unigrams in the Tweets (Relevant)', '50_most_common_unigrams_relevant.png')
    
    # Calculate frequency distribution for bigrams in relevant tweets
    bigrams_relevant = list(zip(all_tokens_relevant, all_tokens_relevant[1:]))
    bigram_freq_relevant = Counter(bigrams_relevant)
    most_common_bigrams_relevant = bigram_freq_relevant.most_common(50)
    plot_most_common(most_common_bigrams_relevant, '50 Most Common Bigrams in the Tweets (Relevant)', '50_most_common_bigrams_relevant.png')
    
    # Calculate frequency distribution for trigrams in relevant tweets
    trigrams_relevant = list(zip(all_tokens_relevant, all_tokens_relevant[1:], all_tokens_relevant[2:]))
    trigram_freq_relevant = Counter(trigrams_relevant)
    most_common_trigrams_relevant = trigram_freq_relevant.most_common(50)
    plot_most_common(most_common_trigrams_relevant, '50 Most Common Trigrams in the Tweets (Relevant)', '50_most_common_trigrams_relevant.png')

if __name__ == "__main__":
    main()
