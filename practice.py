import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer 

sentences = [
    'I love my dog',
    'I love my cat'
]

if __name__ == '__main__':
    tokenizer = Tokenizer(num_words=100) # num_words hyperparameter set to much higher value than necessary
    tokenizer.fit_on_texts(sentences)
    print(tokenizer.word_index)