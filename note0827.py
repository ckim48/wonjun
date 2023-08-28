# NLP --> giving ability to understand the human language
# 1. Text preprocessing --> technique to clean and prepare text data for analysis

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

import re
# Below is for get the text and returns the preprocessed text
# 1. Lowercasing: convert all text to lowercase. This prevents the model from treating
#    'make' and 'Make' as different words.
# 1a.  Remove out the phone numbers
# 1b.  Remove all the punctuations.

# 2. Tokenization: split the text into individual words(tokens).
# 3. Removing Stopwords: removing meaningless words( ex) is,a, the, and,....)

# 4. Lemmatization --> reduce words into original form, ex) running --> run, makes --> make
# 5. Get Final String
def text_preprocessing(text):
    text = text.lower() # Lowercasing -> text preprocessing is an important step.  call 010-1234-1234"

    pattern_phone = r"\d{3}-\d{4}-\d{4}"
    # find the pattern_phone and replace it to "" from the text
    text_no_phone = re.sub(pattern_phone,"",text)

    # ADD A CODE TO REMOVE OUT
    # 1. EMAIL
    # 2. URL
    # 3. EMOJI

    # ^ --> Not
    # \w --> alphabets,digits,and _
    # \s --> white space
    pattern_punc = r"[^\w\s]"
    text_no_punc = re.sub(pattern_punc, "", text_no_phone)

    tokens = word_tokenize(text_no_punc) # Tokenization
    stopwords_list = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stopwords_list]


    lemmatizer = WordNetLemmatizer()
    pos_tags = nltk.pos_tag(filtered_tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(word, get_pos(pos_tag)) for word, pos_tag in pos_tags]

    # list of words to one string
    result = " ".join(lemmatized_tokens)
    return result
def get_pos(word):
    if word.startswith('N'):
        return wordnet.NOUN
    elif word.startswith('V'):
        return wordnet.VERB
    elif word.startswith('J'):
        return wordnet.ADJ
    elif word.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# stopwords_list = set(stopwords.words('english'))
# print(stopwords_list)
sample_text = "Hello, made my name is scott and my phone number is 010-1234-1234"
print(text_preprocessing(sample_text))

# Part of Speech (POS)
# POS tagging on each word using nltk

tokens = word_tokenize(sample_text)
print(tokens)
pos_tags = nltk.pos_tag(tokens)
print(pos_tags)
# NNP --> Proper Noun, singular
# VBZ --> Verb, third person singular present
# NN --> Noun, singular
# JJ --> Adjective
# RB --> Advergb