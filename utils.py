import tensorflow as tf
from scipy.spatial.distance import cosine
import numpy as np
class TweetIterator:
    # Creates an iterator out of a text file
    # Necessary to train tokenizer without loading entire
    # datset into memory.
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = open(file_path, 'r')
    
    def __iter__(self):
        return self
    
    def __next__(self):
        line = self.file.readline()
        if not line:
            self.file.close()
            raise StopIteration
        return line.strip()
    
def get_token_embedding(token, model, tokenizer):
    return model(tokenizer(token, return_tensors="tf"))[0][0][1]

def get_k_nearest_neighbors(
    token, 
    model,
    embeddings, 
    tokenizer, 
    k=10):
    """ get the top k nearest neighbors for a given token
    Args
        token (str): the token to retrieve neighbors for (e.g. "dog")
        model (TFBertForMaskedLM): the BERT model
        embeddings (List[tf.Tensor]): list of tensors representing the embeddings for every word in the vocab
        tokenizer (Tokenizer): associated tokenizer
    Returns
        tuple of (top k neighbors, cosine distances associated w/ top k neighbors)
    
    """
    
    vocab_tokens = list(tokenizer.get_vocab().keys())
    token_embedding = model(tokenizer(token, return_tensors="tf"))[0][0][1]
    
    # get cosine distance of token vs all other tokens
    dists = []
    ignore = tokenizer.all_special_tokens + [token]

    for idx, embedding in enumerate(embeddings):
        if vocab_tokens[idx] in ignore:
            dists.append(10) # don't consider when getting nearest neighbors
        else:
            dists.append(cosine(token_embedding, embedding))

    # sort by cosine distances
    sort_idx = np.array(dists).argsort()
    sorted_dists = np.array(dists)[sort_idx]
    sorted_tokens = np.array(vocab_tokens)[sort_idx]

    return sorted_tokens[:k], sorted_dists[:k]