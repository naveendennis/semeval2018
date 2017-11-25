import numpy as np
import pandas as pd
import math
import nltk
import os.path
import pickle
import gensim
from textblob import Sentence

dir_name = os.path.dirname(os.path.realpath(__file__))
delimiter = '\t'
word_model = None


def load_word_vectors():
    global word_model
    global dir_name
    if not os.path.exists(os.path.join(dir_name, '..', 'output')):
        os.makedirs(os.path.join(dir_name, '..', 'output'))
    model_filename = 'GoogleWord2Vec'
    model_filename = os.path.join(dir_name, '..', 'resources', model_filename)
    if not os.path.exists(model_filename):
        embedding_file_loc = os.path.join(dir_name, '..', 'resources', 'GoogleNews-vectors-negative300.bin')
        print("Loading the data file... Please wait...")
        word_model = gensim.models.KeyedVectors.load_word2vec_format(embedding_file_loc, binary=True)
        print("Successfully loaded 3.6 G bin file!")
        pickle.dump(word_model, open(model_filename, 'wb'))
    else:
        word_model = pickle.load(open(model_filename, 'rb'))
        print('Successfully Loaded the model')


def get_phrase_vector_obj(value):
    return PhraseVector(value)


class PhraseVector:
    def __init__(self, phrase):
        self.phrase = phrase
        self.vector = self.phrase_to_vec(phrase)
        self.pos_tag = self.get_words_in_phrase(phrase)

    @staticmethod
    def convert_vector_set_to_average(vector_set, ignore=[]):
        if len(ignore) == 0:
            return np.mean(vector_set, axis=0)
        else:
            return np.dot(np.transpose(vector_set), ignore) / sum(ignore)

    @staticmethod
    def get_unique_token_tags(vector1, vector2):
        tag_list = []
        for each_tag in vector1.pos_tag + vector2.pos_tag:
            if each_tag not in tag_list:
                tag_list.append(each_tag)
        return tag_list

    def phrase_to_vec(self, phrase):
        # _stop_words = stopwords.words("english")
        phrase = phrase.lower()
        verified_words = [word for word in phrase.split()]
        vector_set = []
        for each_word in verified_words:
            try:
                word_vector = word_model[each_word]
                vector_set.append(word_vector)
            except:
                pass
        return self.convert_vector_set_to_average(vector_set)

    def get_cosine_similarity(self, other_vector):
        cosine_similarity = np.dot(self.vector, other_vector.vector) / (
        np.linalg.norm(self.vector) * np.linalg.norm(other_vector.vector))
        try:
            if math.isnan(cosine_similarity):
                cosine_similarity = 0
        except:
            cosine_similarity = 0
        return cosine_similarity

    def get_words_in_phrase(self, phrase):
        if phrase.strip() == '':
            return []
        else:
            tagged_input = nltk.pos_tag(phrase.split(), tagset='universal')
            prev_item, prev_tag = tagged_input[0]
            g_item_list = [prev_item]
            cur_group_index = 0
            space = ' '
            revised_tag = []
            for cur_item, cur_tag in tagged_input[1:]:
                cur_item = cur_item.lower()
                if prev_tag is cur_tag:
                    g_item_list[cur_group_index] += space + cur_item
                else:
                    revised_tag.append((g_item_list[cur_group_index], prev_tag))
                    prev_tag = cur_tag
                    g_item_list.append(cur_item)
                    cur_group_index += 1
            revised_tag.append((g_item_list[cur_group_index], prev_tag))
            return revised_tag


if __name__ == '__main__':
    train_file_name = os.path.join(dir_name, '..', 'data', 'EI-reg-en_anger_train.txt')
    data = []
    df = pd.read_csv(train_file_name, header=None, delimiter=delimiter)
    load_word_vectors()

    tweet_vectors_obj = None
    tweet_vectors = None
    labels = None
    filename = os.path.join(dir_name, '..', 'resources', 'raw_phrase_vectors_obj')
    if not os.path.exists(filename):
        tweet_vectors_obj = np.vectorize(get_phrase_vector_obj)(df[1].values)
        tweet_vectors = np.array(list(map(lambda x: x.vector, tweet_vectors_obj)))
        labels = df[3].values
        with open(filename, 'wb') as f:
            pickle.dump(tweet_vectors_obj, f)
            pickle.dump(tweet_vectors, f)
            pickle.dump(labels, f)
    else:
        with open(filename, 'rb') as f:
            tweet_vectors_obj = pickle.load(f)
            tweet_vectors = pickle.load(f)
            labels = pickle.load(f)

    polarity_list = []
    subjectivity_list = []
    filename = os.path.join(dir_name, '..', 'resources', 'polarity_and_subjectivity')
    if not os.path.exists(filename):
        polarity_list = np.array(list(map(lambda x: Sentence(x).polarity, df[1].values)))
        subjectivity_list = np.array(list(map(lambda x: Sentence(x).subjectivity, df[1].values)))
        with open(filename, 'wb') as f:
            pickle.dump(polarity_list, f)
            pickle.dump(subjectivity_list, f)
    else:
        with open(filename, 'rb') as f:
            polarity_list = pickle.load(f)
            subjectivity_list = pickle.load(f)

    print(df)
