import pandas as pd
import nltk
nltk.download('vader_lexicon')
from gensim.summarization.summarizer import summarize
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import sys

df = pd.read_csv('C:\\Users\\Sanky27\\OneDrive\\Desktop\\COVIDNEW.csv')
sys.getrecursionlimit()

# we are extracting all th words from each sentence then we are defining the tag of each word
def tag(sentence):
    words = word_tokenize(sentence)
    words = pos_tag(words)
    return words


# we are selecting words having pos tag as noun,verb and adjective
def paraphraseable(tag):
    return tag.startswith('NN') or tag == 'VB' or tag.startswith('JJ')


# assigning noun and verb to the woirdnet
def pos(tag):
    if tag.startswith('NN'):
        return wn.NOUN
    elif tag.startswith('V'):
        return wn.VERB


# it returns the synonym of that word(root word)
def synonyms(word, tag):
    lemma_lists = [ss.lemmas() for ss in wn.synsets(word, pos(tag))]
    lemmas = [lemma.name() for lemma in sum(lemma_lists, [])]
    return set(lemmas)


# this particular function gives the list of synonym of that particular word which is verb,noun.
def synonymIfExists(sentence):
    for (word, t) in tag(sentence):
        if paraphraseable(t):
            syns = synonyms(word, t)
            if syns:
                if len(syns) > 1:
                    yield [word, list(syns)]
                    continue
        yield [word, []]


def paraphrase(sentence):
    return [x for x in synonymIfExists(sentence)]


# function which converts list to string
def listToString(s):
    # initialize an empty string
    str1 = " "

    # return string
    return (str1.join(s))


#
def abstractive_summary(text):
    new_sentence = []

    sentence = sent_tokenize(text)
    for j in sentence:
        final = []
        summarize_list = paraphrase(j)
    #print(summarize_list)
    return summarize_list


# dictionary(word==key,synonym=value)(thats why se created set function to duplicate entries)
def get_key(val, dictionary):
    for key, value in dictionary.items():
        if val == value:
            return key

    return "key doesn't exist"


# it returns the best marching word with the synonym those are availabke for that word.
# wordnet
def get_best_word(word, synonym_list):
    syn1 = wn.synsets(word)[0]
    # print(syn1)
    similarity_matrix = {}
    for i in set(synonym_list):
        # print(i)
        if '_' not in i and '.' not in i:
            syn2 = wn.synsets(i)[0]
            # print(syn1.path_similarity(syn2))
            if syn1 != syn2 and syn1.path_similarity(syn2) != None:
                # print('Inside')
                similarity_matrix[word, i] = syn1.path_similarity(syn2)
            # print(similarity_matrix)
    if len(similarity_matrix) == 0:
        best_word = word
    if len(similarity_matrix) != 0:
        best_word = get_key(max(list(similarity_matrix.values())), similarity_matrix)[1]
        #print(best_word)
    return best_word


def paraphrasing_word_level(original_text):
    modified_text = ''
    summary_word_list = abstractive_summary(original_text)
    best = {}
    for c in summary_word_list:
        filtered_list = []
        #print(len(c[1]))
        if len(c[1]) >= 1:

            for c2 in c[1]:
                if c[0].lower() != c2.lower():
                    filtered_list.append(c2)

            best[c[0]] = get_best_word(str(c[0]), filtered_list)

            # print(filtered_list)
        if len(c[1]) == 0:
            best[c[0]] = c[0]

    modified_text = ' '.join(list(best.values()))
    print(modified_text)
    return modified_text


def get_abstract_summary(original_text):
    errors = []
    results = {}

    try:
        text = original_text

        sentence = sent_tokenize(text)
        whole_sent = []
        for sent in sentence:
            new_text = paraphrasing_word_level(sent)
            whole_sent.append(new_text)
        summ = ' '.join(whole_sent)

        summ_words = summarize(summ, ratio=0.4)

        return summ_words.rstrip().replace('\n', '')
    except:
        return "error"

def sentiment_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    negative=score['neg']
    positive=score['pos']
    if negative > positive:
        print('Negative sentiment')
    elif positive > negative:
        print('Positive sentiment')
    else:
        print('Neutral')
    print(score)

for i in range(10):
    result = get_abstract_summary(df.body[i])
    print('summary=')
    print(result)
    print('######################')
    sentiment_analyse(df.body[i])
    print('####################')
    sentiment_analyse(result)






