from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer, maketrans
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import *
from keras.layers import Flatten
from keras.layers import Embedding
from keras.constraints import min_max_norm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sys
import os
import csv

max_length = -1

def get_rnn_model(vocab_size,
                  embedded_vector_length,
                  embedding_matrix,
                  max_length,
                  optimizer='adam',
                  loss='mean_squared_logarithmic_error',
                  output_activation='relu'):
    model = Sequential()
    e = Embedding(vocab_size, embedded_vector_length, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    lstm = Bidirectional(LSTM(100, activation='relu',
                              kernel_initializer='random_uniform',
                              bias_initializer='zeros',
                              kernel_constraint=min_max_norm(min_value=-1.0, max_value=1.0, rate=0.5, axis=0),
                              bias_constraint=min_max_norm(min_value=-1.0, max_value=1.0, rate=0.5, axis=0)))
    model.add(lstm)
    model.add(Dense(1, activation=output_activation,
                    kernel_initializer='random_uniform',
                    bias_initializer='zeros',
                    kernel_constraint=min_max_norm(min_value=-1.0, max_value=1.0, rate=.5, axis=0),
                    bias_constraint=min_max_norm(min_value=-1.0, max_value=1.0, rate=0.5, axis=0)))
    # compile the model
    model.compile(optimizer=optimizer, loss=loss)
    # summarize the model
    print(model.summary())
    return model
    # fit the model


def get_nn_model(vocab_size,
                  embedded_vector_length,
                  embedding_matrix,
                  max_length,
                  optimizer='adam',
                  loss='mean_squared_logarithmic_error',
                  output_activation='relu'):
    model = Sequential()
    e = Embedding(vocab_size, embedded_vector_length, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation=output_activation))
    # compile the model
    model.compile(optimizer=optimizer, loss=loss)
    # summarize the model
    print(model.summary())
    return model
    # fit the model


def get_embedding_index(embedding_file_name):
    # load the whole embedding into memory
    embeddings_index = dict()
    # f = open('../resources/glove.6B/glove.6B.'+str(embedded_vector_length)+'d.txt')
    with open(embedding_file_name) as f:
        l_no = 0
        for line in f:
            if l_no == 0:
                l_no += 1
                continue
            values = line.split()
            word = values[0]
            try:
                coefs = asarray(values[1:], dtype='float32')
            except ValueError as e:
                print(line)
                print(e)
                coefs = asarray(values[-embedded_vector_length:], dtype='float32')

            embeddings_index[word] = coefs
    return  embeddings_index


def get_padded_docs(docs):
    t = Tokenizer(char_level=False)
    t.fit_on_texts(docs)
    encoded_docs = t.texts_to_sequences(docs)
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    return padded_docs, t

def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    """Converts a text to a sequence of words (or tokens).

    # Arguments
        text: Input text (string).
        filters: Sequence of characters to filter out.
        lower: Whether to convert the input to lowercase.
        split: Sentence split marker (string).

    # Returns
        A list of words (or tokens).
    """
    if lower:
        text = text.lower()

    if sys.version_info < (3,) and isinstance(text, bytes):
        translate_map = dict((ord(c), bytes(split)) for c in filters)
    else:
        translate_map = maketrans(filters, split * len(filters))

    text = text.translate(translate_map)
    seq = text.split(split)
    return [i for i in seq if i]


def write_to_file(tweet_ids, assgn_emotions, tweet_contents, predicted_scores, file_name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(dir_path, '..', 'output')):
        os.makedirs(os.path.join(dir_path, '..', 'output'))
    with open(os.path.join(dir_path, '..', 'output', file_name), 'w') as f:
        file_writer = csv.writer(f, delimiter='\t')
        for each_tweet_id, each_tweet_content, each_emotion, each_score in \
                zip(tweet_ids, tweet_contents, assgn_emotions, predicted_scores):
            file_writer.writerow([each_tweet_id, each_tweet_content, each_emotion, each_score])


def get_docs_contents(file_name):
    global max_length
    docs = []
    labels = []
    tweet_ids = []
    emotions = []
    with open(file_name) as f:
        for each_record in f:
            record_tokens = each_record.split('\t')
            content = record_tokens[1].lower()
            tweet_id = record_tokens[0]
            current_emotion = record_tokens[2]
            docs.append(content)
            emotions.append(current_emotion)
            tweet_ids.append(tweet_id)
            seq = content if char_level else text_to_word_sequence(content)
            temp = len([w for w in seq])
            if temp > max_length:
                max_length = temp
            labels.append(float(record_tokens[3][:-1]))
    return tweet_ids, docs, emotions, labels


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    char_level = False
    docs = []
    labels = []
    tweet_ids = []
    emotions = []

    embedded_vector_length = 300
    emotion_names = ['sadness', 'joy', 'anger', 'sadness']
    embedding_name = 'glove.6B'
    embeddings_index = get_embedding_index(os.path.join(dir_path,
                                                        '..',
                                                        'resources',
                                                        embedding_name,
                                                        'glove.6B.' + str(embedded_vector_length) + 'd.txt'))
    for emotion in emotion_names:
        # emotion = 'sadness'

        tweet_ids, docs, emotions, labels = get_docs_contents(os.path.join(dir_path, '..','data','EI-reg-en_'+emotion+'_train.txt'))

        tweet_ids, docs, emotions, labels = shuffle(tweet_ids, docs, emotions, labels)
        tweet_ids_train, tweet_ids_test, \
        docs_train, doc_test, \
        emotions_train, emotions_test, \
        label_train, label_test = train_test_split(tweet_ids, docs, emotions, labels, test_size=0.33, random_state=42)

        padded_docs_train, t = get_padded_docs(docs_train)
        vocab_size = len(t.word_index) + 1

        print('Loaded %s word vectors.' % len(embeddings_index))
        embedding_matrix = zeros((vocab_size, embedded_vector_length))
        for word, i in t.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        # define model
        model = get_rnn_model(vocab_size, embedded_vector_length, embedding_matrix, max_length)
        model.fit(padded_docs_train, label_train, verbose=1)

        # evaluate the model
        padded_docs_test, _ = get_padded_docs(doc_test)
        predicted_list = model.predict(padded_docs_test)
        write_to_file(tweet_ids_test, emotions_test, doc_test, label_test, emotion + '_' + embedding_name+'_'+'truelabels')
        predicted_list = [each[0] for each in predicted_list]
        write_to_file(tweet_ids_test, emotions_test,  doc_test, predicted_list, emotion+'_'+embedding_name+'_validation')

        # dev set
        tweet_ids, docs, emotions, labels = get_docs_contents(
            os.path.join(dir_path, '..', 'data', 'en_dev', '2018-EI-reg-En-' + emotion + '-dev.txt'))
        padded_docs_dev = get_padded_docs(docs)
        predicted_list = model.predict(padded_docs_dev)
        predicted_list = [each[0] for each in predicted_list]
        write_to_file(tweet_ids, emotions, docs, predicted_list, emotion+'_'+embedding_name+'_dev')

        print('Mean Squared Error of Validation Set: '+str(mean_squared_error(label_test, predicted_list)))