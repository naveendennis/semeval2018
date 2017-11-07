from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers import Embedding
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
# define documents
docs = []
labels = []
max_length = -1
char_level = False
embedded_vector_length = 100
with open('../data/EI-reg-en_anger_train.txt') as f:
    for each_record in f:
        record_tokens = each_record.split('\t')
        content = record_tokens[1].lower()
        docs.append(content)
        seq = content if char_level else text_to_word_sequence(content,
                                                                 '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                                                 True,
                                                                 ' ')
        temp = len([w for w in seq])
        if temp > max_length:
            max_length = temp
        labels.append(float(record_tokens[3][:-1]))

docs, labels = shuffle(docs, labels)
docs_train, doc_test, label_train, label_test = train_test_split(docs,
                                                                 labels,
                                                                 test_size=0.33,
                                                                 random_state=42)

# docs = ['Well done!',
#         'Good work',
#         'Great effort',
#         'nice work',
#         'Excellent!',
#         'Weak',
#         'Poor effort!',
#         'not good',
#         'poor work',
#         'Could have done better.']
# define class labels
# prepare tokenizer

def get_padded_docs(docs):
    t = Tokenizer()
    t.fit_on_texts(docs)
    encoded_docs = t.texts_to_sequences(docs_train)
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    return padded_docs, t

padded_docs_train, t = get_padded_docs(docs_train)
vocab_size = len(t.word_index) + 1
# integer encode the documents

# pad documents to a max length of 4 words

# load the whole embedding into memory
embeddings_index = dict()
f = open('../resources/glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
    # print("Word: "+word+"\n\tEmbedding: "+str(embeddings_index))
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, embedded_vector_length))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
# define model
model = Sequential()
e = Embedding(vocab_size, embedded_vector_length, weights=[embedding_matrix], input_length=max_length, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='softmax'))
# compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs_train, label_train, epochs=2, verbose=0)
# evaluate the model
padded_docs_test = get_padded_docs(doc_test)
loss, accuracy = model.evaluate(padded_docs_test, label_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))