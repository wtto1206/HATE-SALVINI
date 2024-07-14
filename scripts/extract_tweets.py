# Import necessary libraries
import re
import pandas as pd

def read_tweets(file_path):
    """
    Read tweets from a file and return a list of lines.
    
    Args:
    file_path (str): Path to the file containing tweets.
    
    Returns:
    list: List of tweet lines.
    """
    with open(file_path, "r", encoding='utf-8') as f:
        tweets = f.readlines()
    return tweets

def preprocess(line):
    """
    Extract tweet content discarding metadata.
    
    Args:
    line (str): A line from the tweets file.
    
    Returns:
    str: The extracted tweet content or None if no content is found.
    """
    # Extract the text content after <matteosalvinimi>
    results = re.search(r'<matteosalvinimi>(.+)', line)
    if results:
        # Return the text content, which consists of the first group
        return results.group(1).strip()
    else:
        return None

def save_to_csv(tweets, output_file):
    """
    Save tweets to a CSV file.
    
    Args:
    tweets (list): List of tweet strings.
    output_file (str): Path to the output CSV file.
    """
    # Create a dataframe with the tweets
    df = pd.DataFrame(tweets, columns=['tweet'])
    # Save the dataframe to a csv file
    df.to_csv(output_file, index=False)

def main():
    # Define the path to the input file
    path = '/path/to/your/uncleaned_data.txt'  # Replace with your actual path

    # Read in the tweets from the file
    tweets = read_tweets(path)

    # Extract the tweet content discarding metadata
    extracted_tweets = [preprocess(tweet) for tweet in tweets]

    # Combine all tweets into a single document (optional)
    extracted_tweets_as_document = " ".join(extracted_tweets)
    print(extracted_tweets_as_document)

    # Save the extracted tweets to a CSV file
    save_to_csv(extracted_tweets, 'cleaned_data.csv')

if __name__ == "__main__":
    main()
