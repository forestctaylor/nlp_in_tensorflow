import tensorflow as tf

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

if __name__ == '__main__':
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=100) # num_words hyperparameter set to much higher value than necessary
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index 

    sequences = tokenizer.texts_to_sequences(sentences)

    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding="post")
    