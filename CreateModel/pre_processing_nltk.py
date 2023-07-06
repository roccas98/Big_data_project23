from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import download
from collections import defaultdict
from nltk.corpus import wordnet as wn
import contractions


download('averaged_perceptron_tagger')
download("stopwords")
download('punkt')
download('wordnet')

# Remove the contractions from the text
def expand_contractions(text):
    return contractions.fix(text)

# Tokenize the text
def split_into_words(text):
    words = word_tokenize(text)
    return words

# Transform the text in lowercase
def to_lower_case(words):
    words = [word.lower() for word in words]
    return words

# Remove all the non alphabetic words
def keep_alphabetic(words):
    words = [word for word in words if word.isalpha()]
    return words

# Remove the stopwords
def remove_stopwords(words):
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    return words

# Text lemmatization
def lemmatize_words(words):
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word,tag_map[tag[0]]) for word,tag in pos_tag(words)]
    return lemmatized_words

#Function that apply all the previous functions above 
def denoise_text(text):
    text = expand_contractions(text)
    words = split_into_words(text)
    words = to_lower_case(words)
    words = keep_alphabetic(words)
    words = remove_stopwords(words)
    words = lemmatize_words(words)
    return str(words)