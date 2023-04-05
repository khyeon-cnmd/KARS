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
    def __init__(self, save_path, DB_name, mode, text_type, edge_count_type, ngram_range=(1,1)):          
        self.save_path = save_path
        self.DB_name = DB_name
        self.mode = mode
        self.edge_count_type = edge_count_type
        self.text_type = text_type
        # 1. load metadata
        self.metadata_list = []
        with jsonlines.open(f"{save_path}/{DB_name}/{DB_name}.jsonl", 'r') as f:
            for line in f.iter():
                self.metadata_list.append(line)
        self.cv = CountVectorizer(ngram_range=ngram_range, stop_words = None, lowercase=False, tokenizer=lambda x: x.split(' '))
        # 2. load spacy model
        if self.mode == "efficiency":
            self.nlp = spacy.load("en_core_web_sm") #trf
        elif self.mode == "accuracy":
            self.nlp = spacy.load("en_core_web_trf")
        self.nlp.tokenizer = Tokenizer(self.nlp.vocab, token_match=re.compile(r'\S+').match)
        self.node_dict = {}
        self.edge_dict = {}
        self.keyword_extract()
        self.synonym_check()
        self.graph_construct()
        self.save_json()

    def NER(self, word):
        def count_upper(text):
            """
            Returns the number of upper case letters in the given text.
            """
            return sum(1 for c in text if c.isupper())

        def count_element(text):
            """
            Returns the number of elements in the given text.
            """
            elements = {'A': ['As', 'Am', 'Ac', 'At', 'Ar', 'Ag', 'Al', 'Au'], 'B': ['Bk', 'Br', 'Bi', 'Ba', 'Be', 'Bh', 'B'], 'C': ['Cf', 'Cd', 'Cl', 'Cs', 'Co', 'Cn', 'Ce', 'Cu', 'Cr', 'Cm', 'Ca', 'C'], 'D': ['Ds', 'Dy', 'Db'], 'E': ['Er', 'Es', 'Eu'], 'F': ['Fl', 'Fm', 'Fr', 'Fe', 'F'], 'G': ['Ge', 'Gd', 'Ga'], 'H': ['Hg', 'Hs', 'Ho', 'Hf', 'He', 'H'], 'I': ['In', 'Ir', 'I'], 'K': ['Kr', 'K'], 'L': ['Lv', 'Lu', 'La', 'Lr', 'Li'], 'M': ['Mc', 'Md', 'Mg', 'Mt', 'Mo', 'Mn'], 'N': ['Na', 'Nd', 'Ne', 'Np', 'Nh', 'Nb', 'No', 'Ni', 'N'], 'O': ['Os', 'Og', 'O'], 'P': ['Po', 'Pu', 'Pb', 'Pd', 'Pa', 'Pt', 'Pm', 'Pr', 'P'], 'R': ['Rg', 'Ru', 'Ra', 'Rb', 'Re', 'Rf', 'Rn', 'Rh'], 'S': ['Sg', 'Sc', 'Si', 'Sn', 'Sr', 'Sb', 'Se', 'Sm', 'S'], 'T': ['Tb', 'Tc', 'Tm', 'Tl', 'Ts', 'Ta', 'Te', 'Ti', 'Th'], 'U': ['U'], 'V': ['V'], 'W': ['W'], 'X': ['Xe'], 'Y': ['Yb', 'Y'], 'Z': ['Zr', 'Zn']}
            count = 0
            for element in elements.values():
                for e in element:
                    if e in text:
                        count += 1
                        break
            return count

        #if "/" in word:
        #    for w in word.split("/"):
        #        if count_upper(word) < round(len(word)/2):
        #            continue
        #        if count_upper(word) == count_element(word):
        #            return "device"
        if count_upper(word) < round(len(word)/2):
            return "other"
        if count_upper(word) == count_element(word):
            return "material"
        return "other"

    def keyword_extract(self):
        print(f"Loaded number of articles: {len(self.metadata_list)}\n")
        # 1. Year filtering
        remove_list = []
        for metadata in self.metadata_list:
            if not 'published-print' in metadata.keys():
                if not 'published-online' in metadata.keys():
                    remove_list.append(metadata)
        for metadata in remove_list:
            self.metadata_list.remove(metadata)
        print(f"year filtered articles: {len(self.metadata_list)}\n")
        
        # 2. Text_type filtering
        remove_list = []
        if self.text_type == "title":
            for metadata in self.metadata_list:
                if not 'title' in metadata.keys():
                    remove_list.append(metadata)
        elif self.text_type == "abstract":
            for metadata in self.metadata_list:
                if not 'abstract' in metadata.keys():
                    remove_list.append(metadata)
        for metadata in remove_list:
            self.metadata_list.remove(metadata)   
        print(f"text type filtered articles: {len(self.metadata_list)}\n")

        # 3. Text cleaning
        for i, metadata in enumerate(self.metadata_list):
            # 3-0. get text list
            if self.text_type == "title":
                text = metadata[self.text_type][0]
            elif self.text_type == "abstract":
                text = metadata[self.text_type]

            # 3-1. remove any et al written in lower or upper case
            text = text.replace('et al', '').replace('et al.', '').replace('et. al', '').replace('Et. al.', '').replace('Et al', '').replace('Et al.', '').replace('Et. al', '').replace('Et. al.', '')
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
            #text = re.sub(r'[^a-zA-Z0-9/\s]', '', text)
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            # 3-7. remove word which only have number
            text = re.sub(r'\b[0-9]+\b\s*', '', text)
            # 3-8. save text to text_list
            if self.text_type == "title":
                text_list = [text]
            elif self.text_type == "abstract":
                text_list = re.split(r'(?<=[.?!])\s+(?=[A-Z])', text)

            # 4-2. update metadata
            if self.text_type == "title":
                self.metadata_list[i]['title_cleaned'] = text_list
            elif self.text_type == "abstract":
                self.metadata_list[i]['abstract_cleaned'] = text_list

        # 4. Text preprocessing
        print("Text preprocessing...")
        with tqdm(total=len(self.metadata_list)) as pbar:
            for i, metadata in enumerate(self.metadata_list):
                # Load text_list
                if self.text_type == "title":
                    text_list = self.metadata_list[i]['title_cleaned']
                elif self.text_type == "abstract":
                    text_list = self.metadata_list[i]['abstract_cleaned']
                for j, text in enumerate(text_list):
                    # 4-1. Space tokenize -> POS tagging -> number filtering -> lemmatization -> new text
                    new_text = ""
                    doc = self.nlp(text)            
                    for token in doc:
                        if token.pos_ in ["ADJ", "NOUN", "PROPN", "VERB"] and len(token.lemma_) > 1:
                            # 4-2 filter string which only have number
                            if not re.match("^[0-9]+$", str(token)):
                                # 4-3 Make keyword into lemmatized form
                                keyword = token.lemma_
                                # 4-4 check material or device using NER
                                if not keyword == "":
                                    if self.NER(keyword) == "other":
                                        keyword = keyword.lower()
                                    new_text = new_text + " " + keyword

                    #remove first space
                    new_text = new_text[1:]

                    #save new text
                    text_list[j] = new_text

                # 4-2. update metadata
                if self.text_type == "title":
                    self.metadata_list[i]['title_cleaned'] = text_list
                elif self.text_type == "abstract":
                    self.metadata_list[i]['abstract_cleaned'] = text_list

                pbar.update(1)
        print(f"preprocessing text finished\n")

    def synonym_check(self):
        #1. get all keywords
        #2. Upper Lower check
        #3. abbreviation check

        #1. get composition list
        composition_list = []
        for metadata in self.metadata_list:
            if self.text_type == "title":
                text_list = metadata['title_cleaned']
            elif self.text_type == "abstract":
                text_list = metadata['abstract_cleaned']
            
            for text in text_list:
                keywords = text.split(" ")
                for keyword in keywords:
                    #if upper letter exist,
                    if sum(1 for c in keyword if c.isupper()) >= 1:
                        composition_list.append(keyword)

        #2. for all keywords, check upper lower
        with tqdm(total=len(self.metadata_list)) as pbar:
            for i, metadata in enumerate(self.metadata_list):
                if self.text_type == "title":
                    text_list = metadata['title_cleaned']
                elif self.text_type == "abstract":
                    text_list = metadata['abstract_cleaned']

                for j, text in enumerate(text_list):
                    keywords = text.split(" ")
                    for keyword in keywords:
                        for composition in composition_list:
                            if keyword == composition.lower():
                                text = text.replace(keyword, composition)

                    #update text_list
                    text_list[j] = text

                # 4-2. update metadata
                if self.text_type == "title":
                    self.metadata_list[i]['title_cleaned'] = text_list
                elif self.text_type == "abstract":
                    self.metadata_list[i]['abstract_cleaned'] = text_list

                pbar.update(1)
 
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
                    text_list = metadata["title_cleaned"]
                elif self.text_type == "abstract":
                    text_list = metadata["abstract_cleaned"]              

                if self.edge_count_type == "neighbor":
                    # 3-3. Node extraction 
                    for text in text_list:
                        keyword_list = text.split(" ")
                        for keyword in keyword_list:
                            if not keyword == "":
                                # 3-3-1. year freq feature
                                if not keyword in self.node_dict.keys():
                                    self.node_dict[keyword] = {"year":{"total":0}, "NER":None}
                                if not year in self.node_dict[keyword]["year"].keys():
                                    self.node_dict[keyword]["year"][year] = 0

                                self.node_dict[keyword]["year"]["total"] += 1
                                self.node_dict[keyword]["year"][year] += 1

                                # 3-3-2. NER feature
                                self.node_dict[keyword]["NER"] = self.NER(keyword)
                                
                            
                    # 3-4. Edge extraction
                    for text in text_list:
                        keyword_list = text.split(" ")
                        #count only keywords are in nearest neighbor
                        window = 1
                        for i in range(len(keyword_list)):
                            for j in range(i+1, i+window+1):
                                if j < len(keyword_list):
                                    if not keyword_list[i] == "" or not keyword_list[j] == "":
                                        # 3-4-1. check same keyword
                                        if not keyword_list[i] == keyword_list[j]:
                                            #sorting name sequence by alphabet
                                            if keyword_list[i] < keyword_list[j]:
                                                edge_name = f"{keyword_list[i]}-{keyword_list[j]}"
                                            else:
                                                edge_name = f"{keyword_list[j]}-{keyword_list[i]}"
                                            if not edge_name in self.edge_dict.keys():
                                                self.edge_dict[edge_name] = {"year":{"total":0}}
                                            if not year in self.edge_dict[edge_name]["year"].keys():
                                                self.edge_dict[edge_name]["year"][year] = 0
                                            self.edge_dict[edge_name]["year"]["total"] += 1
                                            self.edge_dict[edge_name]["year"][year] += 1

                elif self.edge_count_type == "co-occurrence":
                    # 3-3. Node extraction 
                    for text in text_list:
                        X = self.cv.fit_transform([text])
                        keyword_list = self.cv.get_feature_names_out()
                        for keyword in keyword_list:
                            if not keyword == "":
                                # 3-3-1. year freq feature
                                if not keyword in self.node_dict.keys():
                                    self.node_dict[keyword] = {"year":{"total":0}, "NER":None}
                                if not year in self.node_dict[keyword]["year"].keys():
                                    self.node_dict[keyword]["year"][year] = 0

                                self.node_dict[keyword]["year"]["total"] += 1
                                self.node_dict[keyword]["year"][year] += 1

                                # 3-3-2. NER feature
                                self.node_dict[keyword]["NER"] = self.NER(keyword)
                                
                            
                    # 3-4. Edge extraction
                    for text in text_list:
                        X = self.cv.fit_transform([text])
                        keyword_list = self.cv.get_feature_names_out()
                        Xc = (X.T * X) # this is co-occurrence matrix in sparse csr format
                        Xc.setdiag(0) # sometimes you want to fill same word cooccurence to 0
                        cooccurrences = Xc.todok()
                        for i, j in cooccurrences.keys():
                            if i < j:
                                count = int(cooccurrences[i, j])
                                if not keyword_list[i] == "" or not keyword_list[j] == "":
                                    # 3-4-1. check same keyword
                                    if not keyword_list[i] == keyword_list[j]:
                                        #sorting name sequence by alphabet
                                        if keyword_list[i] < keyword_list[j]:
                                            edge_name = f"{keyword_list[i]}-{keyword_list[j]}"
                                        else:
                                            edge_name = f"{keyword_list[j]}-{keyword_list[i]}"
                                        if not edge_name in self.edge_dict.keys():
                                            self.edge_dict[edge_name] = {"year":{"total":0}}
                                        if not year in self.edge_dict[edge_name]["year"].keys():
                                            self.edge_dict[edge_name]["year"][year] = 0
                                        self.edge_dict[edge_name]["year"]["total"] += count
                                        self.edge_dict[edge_name]["year"][year] += count

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





