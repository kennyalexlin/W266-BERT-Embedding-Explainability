import tensorflow as tf
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
    
def get_token_embedding(tokenizer, embedding_layer, token):
    # convenience function to get the embedding of a particular token
    token_id = tokenizer.convert_tokens_to_ids(token)
    return embedding_layer(tf.constant([[token_id]]))[0][0]