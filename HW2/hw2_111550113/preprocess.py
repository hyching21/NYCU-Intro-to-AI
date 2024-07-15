import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer

import string
from nltk.stem.snowball import SnowballStemmer
import re
def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text


def preprocessing_function(text: str) -> str:
    preprocessed_text = remove_stopwords(text)
    
    # TO-DO 0: Other preprocessing function attemption
    # Begin your code 
    text = text.lower() # lower case
    text = text.replace('<br />',' ')  # remove <br />
    text = re.sub('[^a-zA-Z]', ' ', text) #remove punctuation
    # text = ''.join(char for char in text if (char not in string.punctuation)) #remove punctuation
    text = remove_stopwords(text)
    # stemming
    stemmer = SnowballStemmer(language = "english")
    preprocessed_text = ""
    for word in text.split():
        preprocessed_text += stemmer.stem(word) + ' '
    # End your code

    return preprocessed_text

if __name__ == "__main__":
    s = "One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. <br /><br />The first thing that struck me about Oz was its brutality and unflinching scenes of violence."
    print(preprocessing_function(s))