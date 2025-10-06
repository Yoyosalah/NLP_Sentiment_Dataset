import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin


class RegexCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, list):
            X = pd.Series(X)
        def regex_clean(tweet):
            tweet = re.sub(r"http\S+|www\S+", "", tweet)
            tweet = re.sub(r"@\w+", "", tweet)
            tweet = re.sub(r"#\w+", "", tweet)
            tweet = re.sub(r"[^a-zA-Z\s!?]", "", tweet)
            tweet = re.sub(r"\s+", " ", tweet).strip()
            return tweet
        return X.apply(regex_clean)
    
class TextCleaning(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        import pandas as pd
        if isinstance(X, list):
            X = pd.Series(X)

        def case_stop_stem(text):
            tokens = nltk.word_tokenize(text.lower())
            tokens = [self.stemmer.stem(t) for t in tokens if t not in self.stop_words] 
            return " ".join(tokens)

        return X.apply(case_stop_stem)