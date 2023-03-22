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
    def __init__(self, save_path, DB_name, mode, text_type, ngram_range=(1,1)):          
        self.save_path = save_path
        self.DB_name = DB_name
        self.mode = mode
        self.text_type = text_type
        # 1. load metadata
        self.metadata_list = []
        with jsonlines.open(f"{save_path}/{DB_name}/{DB_name}.jsonl", 'r') as f:
            for line in f.iter():
                self.metadata_list.append(line)
        self.cv = CountVectorizer(ngram_range=ngram_range, stop_words = None, tokenizer=lambda x: x.split(' '))
        # 2. load spacy model
        if self.mode == "efficiency":
            self.nlp = spacy.load("en_core_web_sm") #trf
        elif self.mode == "accuracy":
            self.nlp = spacy.load("en_core_web_trf")
        # 3. set tokenizer
        self.nlp.tokenizer = Tokenizer(self.nlp.vocab, token_match=re.compile(r'\S+').match)
        self.node_dict = {}
        self.edge_dict = {}
        self.keyword_extract()
        self.graph_construct()
        self.save_json()

    def keyword_extract(self):
        print(f"Loaded number of articles: {len(self.metadata_list)}\n")
        # 1. Year filtering
        for metadata in self.metadata_list:
            if not 'published-print' in metadata.keys():
                if not 'published-online' in metadata.keys():
                    self.metadata_list.remove(metadata)
        print(f"year filtered articles: {len(self.metadata_list)}\n")
        
        # 2. Text_type filtering
        if self.text_type == "title":
            for metadata in self.metadata_list:
                if not 'title' in metadata.keys():
                    self.metadata_list.remove(metadata)
        elif self.text_type == "abstract":
            for metadata in self.metadata_list:
                if not 'abstract' in metadata.keys():
                    self.metadata_list.remove(metadata)
        print(f"text type filtered articles: {len(self.metadata_list)}\n")

        # 3. Text cleaning
        for i, metadata in enumerate(self.metadata_list):
            # 3-0. get text
            text = metadata[self.text_type][0]
            # 3-1. latex to text & remove special characters
            text = LatexNodes2Text().latex_to_text(text).replace('\n', '').replace('\r', '').replace('\t', ' ')
            # 3-2. remove from "lt;" to "gt;" matching one by one
            pattern = re.compile(r'lt;.*?gt;')
            text = re.sub(pattern, '', text)
            # 3-3. remove HTML tags
            text = re.sub(r'<.*?>', '', text)
            # 3-4. remove hbox
            text = re.sub(r'hbox', '', text)
            # 3-6. make - to space
            text = re.sub(r'-', ' ', text)
            # 3-5. remove special characters except for /
            text = re.sub(r'[^a-zA-Z0-9/\s]', '', text)
            # 3-7. remove word which only have number
            text = re.sub(r'\b[0-9]+\b\s*', '', text)
            # 3-8. save text
            self.metadata_list[i][self.text_type] = text

        # 4. Text preprocessing
        print("Text preprocessing...")
        with tqdm(total=len(self.metadata_list)) as pbar:
            for i, metadata in enumerate(self.metadata_list):
                # 4-1. Space tokenize -> POS tagging -> number filtering -> lemmatization -> new text
                new_text = ""
                doc = self.nlp(metadata[self.text_type])            
                for token in doc:
                    if token.pos_ in ["ADJ", "NOUN", "PROPN", "VERB"] and len(token.lemma_) > 1:
                        # filter string which only have number
                        if not re.match("^[0-9]+$", str(token)):
                            keyword = token.lemma_.lower()
                            if not keyword == "":
                                new_text = new_text + " " + keyword
                #remove first space
                new_text = new_text[1:]

                # 4-2. update metadata
                if self.text_type == "title":
                    self.metadata_list[i]['title_cleaned'] = new_text
                elif self.text_type == "abstract":
                    self.metadata_list[i]['abstract_cleaned'] = new_text

                pbar.update(1)
        print(f"preprocessing text finished\n")

    def graph_construct(self):
        print("Node & Edge extraction...")
        with tqdm(total=len(self.metadata_list)) as pbar:
            for metadata in self.metadata_list:
                # 3-1. Get year
                if 'published-print' in metadata.keys():
                    year = metadata["published-print"]["date-parts"][0][0]
                elif 'published-online' in metadata.keys():
                    year = metadata["published-online"]["date-parts"][0][0]

                # 3-2. Get text
                if self.text_type == "title":
                    text = metadata["title_cleaned"]
                elif self.text_type == "abstract":
                    text = metadata["abstract_cleaned"]

                # 3-3. Node extraction      
                X = self.cv.fit_transform([text])
                word_count_list = X.toarray().sum(axis=0)
                keyword_list = self.cv.get_feature_names_out()
                for keyword, word_count in zip(keyword_list, word_count_list):
                    if not keyword in self.node_dict.keys():
                        self.node_dict[keyword] = {"total":0}
                    if not year in self.node_dict[keyword]:
                        self.node_dict[keyword][year] = 0
                    self.node_dict[keyword]["total"] += int(word_count)
                    self.node_dict[keyword][year] += int(word_count)

                # 3-4. Edge extraction
                Xc = (X.T * X) # matrix manipulation
                Xc.setdiag(0) # set the diagonals to be zeroes as it's pointless to be 1
                names = self.cv.get_feature_names_out() # This are the entity names (i.e. keywords)
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

    def save_json(self):
        # save node & edge feature
        with open(f"{self.save_path}/{self.DB_name}/node_feature.json", 'w') as f:
            json.dump(self.node_dict, f)
        with open(f"{self.save_path}/{self.DB_name}/edge_feature.json", 'w') as f:
            json.dump(self.edge_dict, f)

        # save cleaned metadata into jsonl file
        with open(f"{self.save_path}/{self.DB_name}/{self.DB_name}_cleaned.jsonl", 'w') as f:
            for i in self.metadata_list: f.write(json.dumps(i) + "\n")





