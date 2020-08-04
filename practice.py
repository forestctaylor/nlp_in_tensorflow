import tensorflow as tf

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

if __name__ == '__main__':
    # num_words hyperparameter set to much higher value than necessary
    # oov_token used in sequence when word does not appear in dictionary
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=100, oov_token="<OOV>") 
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index 

    sequences = tokenizer.texts_to_sequences(['Some other damn sentence', *sentences])

    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding="post")
    print("Dictionary: {}".format(word_index))
    print("Padded seq: {}".format(padded_sequences))