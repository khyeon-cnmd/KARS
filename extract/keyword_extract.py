import os
import json
import re
import spacy
from spacy.tokenizer import Tokenizer
from tqdm import tqdm

class keyword_extraction:
    def __init__(self, DB_path, UPOS_model="efficiency"):
        self.DB_path = f"{DB_path}/database"
        self.UPOS_model = UPOS_model
        if self.UPOS_model == "efficiency":
            self.spacy_model = spacy.load("en_core_web_sm")
        elif self.UPOS_model == "accuracy":
            self.spacy_model = spacy.load("en_core_web_trf")
        self.spacy_model.tokenizer = Tokenizer(self.spacy_model.vocab, token_match=re.compile(r'\S+').match)

    def keyword_tokenization(self):
        print("keyword tokenization")
        # 1. database 내에 있는 모든 paper_index 에 대해서
        for paper_index in tqdm(os.listdir(self.DB_path)):
            # 1-1. KBSE.json 파일이 있는지 확인
            if not os.path.isfile(f"{self.DB_path}/{paper_index}/KBSE.json"):
                continue

            # 1-2. KBSE.json 읽기
            KBSE_dict = json.load(open(f"{self.DB_path}/{paper_index}/KBSE.json", "r", encoding="utf-8-sig"))

            # 1-3. KBSE.json 에 title, published_date 가 없다면 continue
            if not "title" in KBSE_dict["cover_data"].keys() or KBSE_dict["cover_data"]["title"]["0"] == None:
                continue
            if not "published_date" in KBSE_dict["cover_data"].keys() or KBSE_dict["cover_data"]["published_date"] == None:
                continue
            
            # 1-4. KBSE_dict 을 KARS_dict 로 복사
            KARS_dict = {"cover_data": KBSE_dict["cover_data"], "keyword_tokenization":{}}

            # 1-4. title 내 keyword 탐색 및 추출
            title = KARS_dict["cover_data"]["title"]["0"]
            title = title.split()

            # 1-4-1. 약어를 제외한 나머지 단어를 소문자로 변환
            abb_list = []
            for word in title:
                if len(word) == 1 and word.isupper():
                    abb_list.append(word)
                elif len(word) > 1 and len([char for char in word if char.isupper()]) >= 2:
                    abb_list.append(word)
            title = ' '.join([word.lower() if not word in abb_list else word for word in title])

            # 1-4-2. spacy tokenizer 수행
            doc = self.spacy_model(title)

            # 1-4-3. doc token list 를 sentence_token_list 에 저장
            keyword_list = [token for token in doc]

            # 1-4-4. tokenlist 에서 알파벳이 아닌 token을 "[sep]" 으로 대체
            keyword_list = [token if re.sub(r"[a-zA-Z]", "", token.text) == "" else "[sep]" for token in keyword_list]

            # 1-4-5. 명사, 고유명사, 형용사, 대문자가 하나라도 없으면 "[sep]" 으로 대체
            keyword_list = [token if not type(token) == str and (token.pos_ in ["ADJ", "NOUN", "PROPN"] or len([char for char in word if char.isupper()]) > 0) else "[sep]" for token in keyword_list]

            # 1-4-6. stopwords 제거
            keyword_list = [token if not type(token) == str and not token.is_stop else "[sep]" for token in keyword_list]

            # 1-4-7. 모든 token 을 token.text 로 저장
            keyword_list = [token.text if not type(token) == str else token for token in keyword_list]

            # 1-5. "[sep]" 을 제거해 저장
            NER_list = [token for token in keyword_list if not token == "[sep]"]
            
            # 1-6. token list 를 KBSE_dict 에 저장
            KARS_dict["keyword_tokenization"]["title"] = NER_list

            # 1-5. KARS_dict 를 저장
            with open(f"{self.DB_path}/{paper_index}/KARS.json", "w", encoding="utf-8-sig") as f:
                json.dump(KARS_dict, f, indent=4)
