"""CODE TO PERFORM TOPIC MODELING ON SALVINI'S TWEETS"""


#NB: You need to install pytorch 
#!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#!pip install sentence-transformers
#!pip install umap
#!pip install umap-learn
#!pip install hdbscan
#!pip install umap-learn
#!pip install -U feel-it



# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import umap.umap_ as umap
from sentence_transformers import SentenceTransformer
import hdbscan
from wordcloud import WordCloud

# Function to load data
def load_data(file_path):
    """
    Load cleaned tweets data from CSV file.
    
    Args:
    file_path (str): Path to the CSV file containing cleaned tweets.
    
    Returns:
    pd.DataFrame: DataFrame containing tweet data.
    """
    df = pd.read_csv(file_path)
    return df

# Function to create sentence embeddings
def create_embeddings(data, save_path='embeddings.npy'):
    """
    Create sentence embeddings using SentenceTransformer model.
    
    Args:
    data (list): List of tweet texts.
    save_path (str): Path to save embeddings file.
    
    Returns:
    np.ndarray: Array of sentence embeddings.
    """
    model = SentenceTransformer('distilbert-multilingual-nli-stsb-quora-ranking')
    embeddings = model.encode(data, show_progress_bar=True)
    np.save(save_path, embeddings)   # Save the embeddings to a file
    return embeddings

# Function for UMAP dimensionality reduction
def dimensionality_reduction(embeddings, n_neighbors=15, n_components=5, metric='cosine', random_seed=42):
    """
    Perform UMAP dimensionality reduction on embeddings.
    
    Args:
    embeddings (np.ndarray): Array of sentence embeddings.
    n_neighbors (int): Number of neighbors for UMAP.
    n_components (int): Number of components for UMAP.
    metric (str): Metric for UMAP.
    random_seed (int): Random seed for reproducibility.
    
    Returns:
    np.ndarray: UMAP-transformed embeddings.
    """
    umap_embeddings = umap.UMAP(n_neighbors=n_neighbors, 
                                n_components=n_components, 
                                metric=metric,
                                random_state=random_seed).fit_transform(embeddings)
    return umap_embeddings

# Function for clustering using HDBSCAN
def clustering_umap(embeddings, min_cluster_size=15, metric='euclidean', cluster_selection_method='eom'):
    """
    Perform clustering using HDBSCAN on UMAP-transformed embeddings.
    
    Args:
    embeddings (np.ndarray): Array of UMAP-transformed embeddings.
    min_cluster_size (int): Minimum cluster size for HDBSCAN.
    metric (str): Metric for HDBSCAN.
    cluster_selection_method (str): Cluster selection method for HDBSCAN.
    
    Returns:
    hdbscan.HDBSCAN: HDBSCAN clustering model.
    """
    cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                              metric=metric,                      
                              cluster_selection_method=cluster_selection_method).fit(embeddings)
    return cluster

# Function for computing c-TF-IDF
def c_tf_idf(documents, m, ngram_range=(1, 1)):
    """
    Compute c-TF-IDF for documents.
    
    Args:
    documents (list): List of document texts.
    m (int): Total number of documents.
    ngram_range (tuple): Range for n-grams.
    
    Returns:
    np.ndarray, CountVectorizer: c-TF-IDF matrix and fitted CountVectorizer.
    """
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)
    return tf_idf, count

# Function to extract top words per topic using c-TF-IDF
def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    """
    Extract top n words per topic using c-TF-IDF scores.
    
    Args:
    tf_idf (np.ndarray): c-TF-IDF matrix.
    count (CountVectorizer): Fitted CountVectorizer.
    docs_per_topic (pd.DataFrame): DataFrame with document per topic.
    n (int): Number of top words to extract per topic.
    
    Returns:
    dict: Dictionary of top n words per topic.
    """
    words = count.get_feature_names_out()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words

# Function to extract topic sizes
def extract_topic_sizes(df):
    """
    Extract topic sizes from dataframe.
    
    Args:
    df (pd.DataFrame): DataFrame with topics and document counts.
    
    Returns:
    pd.DataFrame: DataFrame with topic sizes sorted in descending order.
    """
    topic_sizes = df.groupby(['Topic']).Doc.count().reset_index().rename(columns={"Doc": "Size"}).sort_values("Size", ascending=False)
    return topic_sizes

# Function to plot the most frequent topics
def plot_topic_sizes(topic_sizes):
    """
    Plot the most frequent topics.
    
    Args:
    topic_sizes (pd.DataFrame): DataFrame with topic sizes.
    """
    filtered_topic_sizes = topic_sizes[topic_sizes['Topic'] != -1].head(20)
    colors = ['#545E75' if topic not in [122, 64, 81, 123] else 'orange' for topic in filtered_topic_sizes['Topic']]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    bars = filtered_topic_sizes.plot(kind='bar', x='Topic', y='Size', ax=ax, color=colors, legend=False)
    
    x_coords = [bar.get_x() + bar.get_width() / 2 for bar in bars.patches]
    y_coords = [bar.get_height() for bar in bars.patches]
    ax.plot(x_coords, y_coords, 'k-', linewidth=1)
    
    custom_lines = [plt.Line2D([0], [0], color='orange', lw=4),
                    plt.Line2D([0], [0], color='#545E75', lw=4)]
    ax.legend(custom_lines, ['Immigration-related topics', 'Other topics'])
    
    ax.set_ylabel('Number of Documents')
    ax.set_title('Most Frequent 20 Topics in Salvini\'s Tweets')
    
    plt.show()
    fig.savefig('topics.png')

# Function to create and save WordCloud for a specific topic
def create_wordcloud(top_n_words, topic_id):
    """
    Create and save WordCloud representation for a specific topic.
    
    Args:
    top_n_words (dict): Dictionary of top words per topic.
    topic_id (int): Topic ID for which WordCloud is to be created.
    """
    top_words = top_n_words[topic_id]  # Replace with your actual topic number
    words = [word for word, _ in top_words]
    tfidf_scores = [score for _, score in top_words]
    
    word_dict = {word: score for word, score in zip(words, tfidf_scores)}
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_dict)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'WordCloud for Topic {topic_id}')
    plt.savefig(f'wordcloud_topic_{topic_id}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Function to extract tweets in a specific topic
def extract_topic_tweets(df, topic_id):
    """
    Extract tweets belonging to a specific topic.
    
    Args:
    df (pd.DataFrame): DataFrame containing tweets and their topics.
    topic_id (int): Topic ID to filter tweets.
    
    Returns:
    pd.DataFrame: DataFrame containing tweets for the specified topic.
    """
    topic_tweets = df[df['Topic'] == topic_id]
    return topic_tweets

def main():
    # Load cleaned data
    df = load_data('cleaned_data.csv')  # Replace with your actual path
    
    # Extract tweet texts
    data = list(df['tweet'])
    
    # Load pre-generated embeddings
    embeddings = np.load('embeddings.npy')
    
    # Perform UMAP dimensionality reduction
    umap_embeddings = dimensionality_reduction(embeddings)
    
    # Perform HDBSCAN clustering
    cluster = clustering_umap(umap_embeddings)
    
    # Create dataframe with topics and document counts
    df['Topic'] = cluster.labels_
    df['Doc_ID'] = range(len(df))
    docs_per_topic = df.groupby(['Topic'], as_index=False).agg({'tweet': ' '.join})
    
    # Compute c-TF-IDF
    tf_idf, count = c_tf_idf(docs_per_topic['tweet'].values, m=len(data))
    
    # Extract top words per topic
    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic)
    
    # Extract topic sizes
    topic_sizes = extract_topic_sizes(df)
    
    # Plot the most frequent topics
    plot_topic_sizes(topic_sizes)
    
    # Create and save WordCloud for immigration-related topics
    for topic_id in [122, 64, 81, 123]:  # Replace with actual topic numbers
        create_wordcloud(top_n_words, topic_id)
    
    # Example: Extract all tweets in a specific topic (Topic 122)
    topic_122 = extract_topic_tweets(df, 122)
    topic_122.to_csv('topic_122.csv', index=False)

    # Example: Extract all tweets in a specific topic (Topic 64)
    topic_64 = extract_topic_tweets(df, 64)
    topic_64.to_csv('topic_64.csv', index=False)

    # Example: Extract all tweets in a specific topic (Topic 81)
    topic_81 = extract_topic_tweets(df, 81)
    topic_81.to_csv('topic_81.csv', index=False)

    # Example: Extract all tweets in a specific topic (Topic 123)
    topic_123 = extract_topic_tweets(df, 123)
    topic_123.to_csv('topic_123.csv', index=False)

    # Save all topics to a CSV file
    all_topics = pd.concat([topic_122, topic_64, topic_81, topic_123])
    all_topics.to_csv('all_topics.csv', index=False)

if __name__ == "__main__":
    main()
