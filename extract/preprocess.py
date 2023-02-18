
from datetime import date
from pylatexenc.latex2text import LatexNodes2Text

class preprocess:
    def __init__(self, df):
        self.df = df
        self.df['date'] = self.df['date']

    def 