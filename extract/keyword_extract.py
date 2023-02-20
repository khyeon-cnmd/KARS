import spacy
import jsonlines
import re
from tqdm import tqdm
import json
import pandas as pd
import warnings
# ignore warning of sklearn
warnings.filterwarnings(action="ignore", category=UserWarning)
from pylatexenc.latex2text import LatexNodes2Text
from sklearn.feature_extraction.text import CountVectorizer
from spacy.tokenizer import Tokenizer

class keyword_extract:
    def __init__(self, save_path, DB_name, mode, text_type):          
        self.save_path = save_path
        self.DB_name = DB_name
        self.mode = mode
        self.text_type = text_type
        self.metadata_list = []
        with jsonlines.open(f"{save_path}/{DB_name}/{DB_name}.jsonl", 'r') as f:
            for line in f.iter():
                self.metadata_list.append(line)
        self.cv = CountVectorizer(ngram_range=(1,1), stop_words = None, tokenizer=lambda x: x.split(' '))
        self.node_dict = {}
        self.edge_dict = {}
        self.tokenize()

    def tokenize(self):
        # 1. load spacy model
        if self.mode == "efficiency":
            nlp = spacy.load("en_core_web_sm") #trf
        elif self.mode == "accuracy":
            nlp = spacy.load("en_core_web_trf")

        # 2. Use only space tokenizer
        nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)

        # 3. load text and preprocessing spectial characters
        with tqdm(total=len(self.metadata_list)) as pbar:
            for metadata in self.metadata_list:
                # 3-1. Get metadatas
                if 'published-print' in metadata.keys():
                    year = str(metadata['published-print']["date-parts"][0][0])
                elif 'published-online' in metadata.keys():
                    year = str(metadata['published-online']["date-parts"][0][0])
                else:
                    continue

                # 3-2. text preprocessing
                if self.text_type == "title":
                    text = LatexNodes2Text().latex_to_text(str(metadata['title'])).replace('\n', '').replace('\r', '').replace('\t', ' ')
                if self.text_type == "abstract":
                    if not 'abstract' in metadata.keys():
                        continue
                    text = LatexNodes2Text().latex_to_text(str(metadata['abstract'])).replace('\n', '').replace('\r', '').replace('\t', ' ')
                text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

                # 3-3. Node extraction
                new_text = ""
                doc = nlp(text)            
                for token in doc:
                    if token.pos_ in ["ADJ", "NOUN", "PROPN", "VERB"] and len(token.lemma_) > 1:
                        has_number = lambda stringVal: any(char.isdigit() for char in stringVal)
                        if not has_number(str(token)):
                            keyword = token.lemma_.lower()
                            if not keyword == "":
                                if not keyword in self.node_dict.keys():
                                    self.node_dict[keyword] = {"total":0}
                                if not year in self.node_dict[keyword]:
                                    self.node_dict[keyword][year] = 0
                                self.node_dict[keyword]["total"] += 1
                                self.node_dict[keyword][year] += 1
                                new_text = new_text + " " + keyword

                # 3-4. Edge extraction
                X = self.cv.fit_transform([new_text])
                Xc = (X.T * X) # matrix manipulation
                Xc.setdiag(0) # set the diagonals to be zeroes as it's pointless to be 1
                names = self.cv.get_feature_names_out() # This are the entity names (i.e. keywords)
                df = pd.DataFrame(data = Xc.toarray(), columns = names, index = names)
                for i in range(len(names)):
                    for j in range(len(names)):
                        if i < j:
                            if not names[i] == "" and not names[j] == "":
                                #sorting name sequence by alphabet
                                if names[i] < names[j]:
                                    edge_name = f"{names[i]}-{names[j]}"
                                else:
                                    edge_name = f"{names[j]}-{names[i]}"
                                if not edge_name in self.edge_dict.keys():
                                    self.edge_dict[edge_name] = {"total":0}
                                if not year in self.edge_dict[edge_name]:
                                    self.edge_dict[edge_name][year] = 0
                                self.edge_dict[edge_name]["total"] += 1
                                self.edge_dict[edge_name][year] += 1
                pbar.update(1)

        # 4. save dict to json
        with open(f"{self.save_path}/{self.DB_name}/node_feature.json", 'w') as f:
            json.dump(self.node_dict, f)
        with open(f"{self.save_path}/{self.DB_name}/edge_feature.json", 'w') as f:
            json.dump(self.edge_dict, f)







