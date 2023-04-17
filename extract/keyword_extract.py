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
from spacy.tokenizer import Tokenizer

class keyword_extract:
    def __init__(self, save_path, DB_name, tokenizer_type):          
        self.save_path = save_path
        self.DB_name = DB_name
        self.tokenizer_type = tokenizer_type
        # 1. load metadata
        self.metadata_list = []
        with jsonlines.open(f"{self.save_path}/Article_collection/{self.DB_name}.jsonl", 'r') as f:
            for line in f.iter():
                self.metadata_list.append(line)
        print(f"Loaded number of articles: {len(self.metadata_list)}")
        # 2. load spacy tokenizer_typel
        if self.tokenizer_type == "efficiency":
            self.nlp = spacy.load("en_core_web_sm")
        elif self.tokenizer_type == "accuracy":
            self.nlp = spacy.load("en_core_web_trf")
        self.nlp.tokenizer = Tokenizer(self.nlp.vocab, token_match=re.compile(r'\S+').match)
        self.metadata_filtering()
        self.text_cleansing()
        self.text_filtering()
        self.synonym_checking()
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

    def metadata_filtering(self):
        print(f"Articles filtering...")
        # 1. Year filtering
        remove_list = []
        for metadata in self.metadata_list:
            if not 'published-print' in metadata.keys():
                if not 'published-online' in metadata.keys():
                    remove_list.append(metadata)
        for metadata in remove_list:
            self.metadata_list.remove(metadata)
        print(f"year filtered articles: {len(self.metadata_list)}")
        
        # 2. Text_type filtering
        remove_list = []
        for metadata in self.metadata_list:
            if not 'title' in metadata.keys() and not 'abstract' in metadata.keys():
                remove_list.append(metadata)
        for metadata in remove_list:
            self.metadata_list.remove(metadata)   
        print(f"text type filtered articles: {len(self.metadata_list)}")

    def text_cleansing(self):
        def regular_expression(text):
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

            return text

        print("Text cleansing...")
        with tqdm(total=len(self.metadata_list)) as pbar:
            for i, metadata in enumerate(self.metadata_list):
                if "title" in metadata.keys():
                    text = metadata["title"][0]         
                    text = regular_expression(text)
                    text_list = [text]
                    self.metadata_list[i]['title_cleaned'] = text_list

                if "abstract" in metadata.keys():
                    text = metadata["abstract"]
                    text = regular_expression(text)
                    text_list = re.split(r'(?<=[.?!])\s+(?=[A-Z])', text)
                    self.metadata_list[i]['abstract_cleaned'] = text_list

                pbar.update(1)

    def text_filtering(self):
        def NLP_tokenizer(text):
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

            return new_text

        print("Text filtering...")
        with tqdm(total=len(self.metadata_list)) as pbar:
            for i, metadata in enumerate(self.metadata_list):
                if "title_cleaned" in metadata.keys():
                    text_list = self.metadata_list[i]['title_cleaned']
                    for j, text in enumerate(text_list):
                        new_text = NLP_tokenizer(text)
                        text_list[j] = new_text
                    self.metadata_list[i]['title_cleaned'] = text_list

                if "abstract_cleaned" in metadata.keys():
                    text_list = self.metadata_list[i]['abstract_cleaned']
                    for j, text in enumerate(text_list):
                        new_text = NLP_tokenizer(text)
                        text_list[j] = new_text
                    self.metadata_list[i]['abstract_cleaned'] = text_list

                pbar.update(1)

    def synonym_checking(self):
        #1. get all keywords
        #2. Upper Lower check
        #3. abbreviation check

        #1. get composition list
        print("Composition list making...")
        composition_list = []
        for metadata in self.metadata_list:
            text_list = []
            if "title" in metadata.keys():
                text_list += metadata['title_cleaned']
            if "abstract" in metadata.keys():
                text_list += metadata['abstract_cleaned']
            
            for text in text_list:
                keywords = text.split(" ")
                for keyword in keywords:
                    #if upper letter exist,
                    if sum(1 for c in keyword if c.isupper()) >= 1:
                        composition_list.append(keyword)
        print(f"Composition list\n{composition_list}")

        #2. for all keywords, check upper lower
        print("Synonym checking...")
        with tqdm(total=len(self.metadata_list)) as pbar:
            for i, metadata in enumerate(self.metadata_list):
                if "title" in metadata.keys():
                    text_list = metadata['title_cleaned']
                    for j, text in enumerate(text_list):
                        keywords = text.split(" ")
                        for keyword in keywords:
                            for composition in composition_list:
                                if keyword == composition.lower():
                                    text = text.replace(keyword, composition)

                        #update text_list
                        text_list[j] = text
                    self.metadata_list[i]['title_cleaned'] = text_list

                if "abstract" in metadata.keys():
                    text_list = metadata['abstract_cleaned']
                    for j, text in enumerate(text_list):
                        keywords = text.split(" ")
                        for keyword in keywords:
                            for composition in composition_list:
                                if keyword == composition.lower():
                                    text = text.replace(keyword, composition)

                        #update text_list
                        text_list[j] = text
                    self.metadata_list[i]['abstract_cleaned'] = text_list

                pbar.update(1)
 
    def save_json(self):
        # save cleaned metadata into jsonl file
        with open(f"{self.save_path}/Keyword_extraction/{self.DB_name}_cleaned.jsonl", 'w') as f:
            for i in self.metadata_list: f.write(json.dumps(i) + "\n")

        #2. save search results as csv
        df = pd.DataFrame(columns=["Index","Title","Abstract","Published date","DOI"])
        for metadata in self.metadata_list:
            if "published-print" in metadata.keys():
                published_date = metadata["published-print"]["date-parts"][0]
            elif "published-online" in metadata.keys():
                published_date = metadata["published-online"]["date-parts"][0]
            if not "title_cleaned" in metadata.keys():
                metadata["title_cleaned"] = ""
            if not "abstract_cleaned" in metadata.keys():
                metadata["abstract_cleaned"] = ""
            df = df.append({"Index":len(df),"Title":metadata["title_cleaned"][0],"Abstract":metadata["abstract_cleaned"],"Published date":published_date,"DOI":metadata["DOI"]},ignore_index=True)
        df.to_csv(f"{self.save_path}/Keyword_extraction/{self.DB_name}_cleaned.csv", index=False)




