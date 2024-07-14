import pandas as pd
        
        
def get_tweet_list(path):
    """Loads tweets from a text file.
    
    Args:
        path (str): The file path of the text file.
        
    Returns:
        list: a list of all lines in the text file
    """
    with open(path, 'r') as file:
        tweets = file.readlines()
    return tweets

def get_tweet_iterator(path):
    """Loads tweets from a text file.
    
    Args:
        path (str): The file path of the text file.
        
    Returns:
        iterator: an iterator over the lines in the text file
    """
    with open(path, 'r') as file:
        for line in file:
            yield line