import os
import json
import spacy 
import networkx as nx
from spacy.tokenizer import Tokenizer
import re
from tqdm import tqdm
import scipy as sp
import scipy.io  # for mmread() and mmwrite()
import io  # Use BytesIO as a stand-in for a Python file object

class PSPP_network:
    def __init__(self, DB_path):
        self.DB_path = DB_path
        if os.path.isdir(self.DB_path + "/KARS") == False:
            os.mkdir(self.DB_path + "/KARS")
        self.spacy_model = spacy.load("en_core_sci_sm")
        self.spacy_model.tokenizer = Tokenizer(self.spacy_model.vocab, token_match=re.compile(r'\S+').match)

    def PSPP_relationship(self, file_type, para_type):
        print("PSPP_relationship")
        if file_type == "KBSE":
            data_loc = "cover_data"
        elif file_type == "DEFT":
            data_loc = "text_data"

        for paper_index in tqdm(os.listdir(f"{self.DB_path}/database")):
            if os.path.isfile(f"{self.DB_path}/database/{paper_index}/{file_type}.json"):
                # read json to dict
                text_dict = json.load(open(f"{self.DB_path}/database/{paper_index}/{file_type}.json", "r", encoding="utf-8-sig"))

                # if it is not major cluster paper, pass
                if data_loc == "cover_data" and text_dict[data_loc]["cluster_label"] != "-1":
                    continue
                
                # if there is no published_date, pass
                # if not text_dict[data_loc]["published_date"]:
                #     continue
                # else:
                #     if isinstance(text_dict[data_loc]["published_date"][0], list):
                #         year = text_dict[data_loc]["published_date"][0][0]
                #     else:
                #         year = text_dict[data_loc]["published_date"][0]
                    
                #     if len(str(year)) != 4:
                #         continue

                # if there is no para_type, pass
                if not para_type in text_dict[data_loc].keys() or text_dict[data_loc][para_type] == None:
                    continue

                # merge all paragraph into one string
                text = ""
                for para_index in text_dict[data_loc][para_type].keys():
                    text += text_dict[data_loc][para_type][para_index]
                if text == "":
                    continue

                # split string into sentence
                sentence_list = text.split(". ")

                # make network
                G = nx.DiGraph()

                # Make relationship
                PSPP_relationship = {}
                for i, sen in enumerate(sentence_list):
                    PSPP_relationship[i] = {}
                    doc = self.spacy_model(sen)

                    # 1. find all keyword list
                    index = 0 
                    keyword_list = []
                    while index < len(doc):
                        token = doc[index]
                        if token.pos_ in ["ADJ", "NOUN", "PROPN"]: # 명사, 형용사면 Keyword start
                            keyword = []
                            for next_token in doc[index:]:
                                next_lemma = next_token.lemma_
                                next_pos = next_token.pos_
                                # 2. find end word  
                                if re.sub(r"[^0-9]", "", next_lemma) != "": # 숫자가 1글자라도 포함되어 있으면 끝
                                    break
                                if "(" in next_lemma or ")" in next_lemma: # 괄호면 끝
                                    break
                                if "." in next_lemma or "," in next_lemma or ";" in next_lemma or ":" in next_lemma: # . , ;, : 이면 끝
                                    keyword.append(next_token)
                                    break
                                if re.sub(r"[a-zA-Z]", "", next_lemma) != "": # 단어에 특수문자가 하나라도 있으면 끝
                                    break
                                if next_pos not in ["ADJ", "NOUN", "PROPN"]: # 명사, 형용사가 아니면 끝
                                    break
                                keyword.append(next_token)
                                index += 1
                            if keyword:
                                keyword_list.append(keyword)
                        index += 1

                    # 2. find relationship of the root word
                    root_connected_list = []
                    for i, keyword in enumerate(keyword_list):
                        root_word = keyword[-1]
                        child_word = root_word

                        while True:
                            root_word2 = child_word.head
                            connection = False
                            for j, keyword2 in enumerate(keyword_list):
                                if root_word2 == keyword2[-1]:
                                    root_word_lemma = root_word.lemma_.replace(".", "").replace(",", "").replace(";", "").replace(":", "").lower()
                                    root_word2_lemma = root_word2.lemma_.replace(".", "").replace(",", "").replace(";", "").replace(":", "").lower()
                                    # root_word_lemma = ' '.join([token.lemma_.replace(".", "").replace(",", "").replace(";", "").replace(":", "") for token in keyword])
                                    # root_word2_lemma = ' '.join([token.lemma_.replace(".", "").replace(",", "").replace(";", "").replace(":", "") for token in keyword2])

                                    if not G.has_edge(root_word_lemma, root_word2_lemma):
                                        G.add_edge(root_word_lemma, root_word2_lemma, weight=1)
                                        #G[root_word_lemma][root_word2_lemma][str(paper_index)] = 1
                                    else:
                                        G[root_word_lemma][root_word2_lemma]["weight"] += 1
                                        #G[root_word_lemma][root_word2_lemma][str(paper_index)] += 1
                                    connection = True
                                    break

                            # 4. root_word 의 부모가 ROOT 인지 확인
                            if connection == False and root_word2.dep_ == "ROOT":
                                #print(root_word, "->", root_word.head, "(ROOT)")
                                root_connected_list.append(keyword)
                                connection = True
                                break

                            if connection == False:
                                child_word = root_word2
                            else:
                                break
                         

                    # 3. root 에 연결된 token에 대해서 의존관계 재정립
                    new_root = None
                    for keyword in root_connected_list:
                        root_word = keyword[-1]
                        if root_word.dep_ == "nsubjpass":  #수동태 주어 우선
                            new_root = keyword
                            break
                        elif root_word.dep_ == "nsub": # 능동태 주어 제외
                            pass
                        else: # -> 목적어 우선
                            new_root = keyword
                            break

                    # 4. 새로운 root 에 연결된 token에 대해서 의존관계 재정립
                    for keyword in root_connected_list:
                        if not keyword == new_root:
                            root_word = keyword[-1]
                            root_word_lemma = root_word.lemma_.replace(".", "").replace(",", "").replace(";", "").replace(":", "").lower()
                            new_root_lemma = new_root[-1].lemma_.replace(".", "").replace(",", "").replace(";", "").replace(":", "").lower()
                            # root_word_lemma = ' '.join([token.lemma_.replace(".", "").replace(",", "").replace(";", "").replace(":", "") for token in keyword])
                            # new_root_lemma = ' '.join([token.lemma_.replace(".", "").replace(",", "").replace(";", "").replace(":", "") for token in new_root])
                            if not G.has_edge(root_word_lemma, new_root_lemma):
                                G.add_edge(root_word_lemma, new_root_lemma, weight=1)
                                #G[root_word_lemma][new_root_lemma][str(paper_index)] = 1
                            else:
                                G[root_word_lemma][new_root_lemma]["weight"] += 1
                                #G[root_word_lemma][new_root_lemma][str(paper_index)] += 1
                
                    # add node weight
                    for keyword in keyword_list:
                        node = keyword[-1].lemma_.replace(".", "").replace(",", "").replace(";", "").replace(":", "").lower()
                        # node = ' '.join([token.lemma_.replace(".", "").replace(",", "").replace(";", "").replace(":", "") for token in keyword])
                        if node in G.nodes:
                            if not "weight" in G.nodes[node]:
                                G.nodes[node]["weight"] = 1
                            else:
                                G.nodes[node]["weight"] += 1

                            # if not str(year) in G.nodes[node]:
                            #     G.nodes[node][str(year)] = 1
                            # else:
                            #     G.nodes[node][str(year)] += 1

                # save network
                nx.write_gexf(G, f"{self.DB_path}/database/{paper_index}/PSPP_{para_type}.gexf")

    def PSPP_co_occurrence(self):
        print("PSPP_co_occurrence")
        for paper_index in tqdm(os.listdir(f"{self.DB_path}/database")):
            if os.path.isfile(f"{self.DB_path}/database/{paper_index}/KBSE.json"):
                # read json to dict
                text_dict = json.load(open(f"{self.DB_path}/database/{paper_index}/KBSE.json", "r", encoding="utf-8-sig"))
                
                # if it is not major cluster paper, pass
                if text_dict["cover_data"]["cluster_label"] != "-1":
                    continue
                
                # if there is no published_date, pass
                if not text_dict["cover_data"]["published_date"]:
                    continue
                else:
                    if isinstance(text_dict["cover_data"]["published_date"][0], list):
                        year = text_dict["cover_data"]["published_date"][0][0]
                    else:
                        year = text_dict["cover_data"]["published_date"][0]
                    
                    if len(str(year)) != 4:
                        continue

                # if there is no para_type, pass
                if not "title" in text_dict["cover_data"].keys() or text_dict["cover_data"]["title"] == None:
                    continue
                text = text_dict["cover_data"]["title"]["0"]
                
                # make network
                G = nx.Graph()

                # Make relationship
                doc = self.spacy_model(text)

                # 1. find all keyword list
                index = 0 
                keyword_list = []
                while index < len(doc):
                    token = doc[index]
                    if token.pos_ in ["ADJ", "NOUN", "PROPN"]: # 명사, 형용사면 Keyword start
                        keyword = []
                        for next_token in doc[index:]:
                            next_lemma = next_token.lemma_
                            next_pos = next_token.pos_
                            # 2. find end word  
                            if re.sub(r"[^0-9]", "", next_lemma) != "": # 숫자가 1글자라도 포함되어 있으면 끝
                                break
                            if "(" in next_lemma or ")" in next_lemma: # 괄호면 끝
                                break
                            if "." in next_lemma or "," in next_lemma or ";" in next_lemma or ":" in next_lemma: # . , ;, : 이면 끝
                                keyword.append(next_token)
                                break
                            if re.sub(r"[a-zA-Z]", "", next_lemma) != "": # 단어에 특수문자가 하나라도 있으면 끝
                                break
                            if next_pos not in ["ADJ", "NOUN", "PROPN"]: # 명사, 형용사가 아니면 끝
                                break
                            keyword.append(next_token)
                            index += 1
                        if keyword:
                            keyword_list += [keyword.lemma_.replace(".","").replace(",","").replace(";","").replace(":","").lower() for keyword in keyword]
                    index += 1
                
                # 2. get edge of co-occurrence
                for i, keyword in enumerate(keyword_list):
                    for j, keyword2 in enumerate(keyword_list):
                        if i <= j:
                            continue
                        if G.has_edge(keyword, keyword2):
                            G[keyword][keyword2]["weight"] += 1
                        else:
                            G.add_edge(keyword, keyword2, weight=1)

                # add node weight
                for keyword in keyword_list:
                    if not keyword in G.nodes:
                        G.add_node(keyword)

                    if "weight" in G.nodes[keyword]:
                        G.nodes[keyword]["weight"] += 1
                    else:
                        G.nodes[keyword]["weight"] = 1
                    
                    if str(year) in G.nodes[keyword]:
                        G.nodes[keyword][str(year)] += 1
                    else:
                        G.nodes[keyword][str(year)] = 1

                # save network
                nx.write_gexf(G, f"{self.DB_path}/database/{paper_index}/PSPP_title.gexf")

    def construct_PSPP_network(self, edge_type, para_type):
        print(f"Construct PSPP network")
        if edge_type == "relationship":
            Total_G = nx.DiGraph()
        elif edge_type == "co_occurrence":
            Total_G = nx.Graph()

        for paper_index in tqdm(os.listdir(f"{self.DB_path}/database")):
            if os.path.isfile(f"{self.DB_path}/database/{paper_index}/PSPP_{para_type}.gexf"):
                G = nx.read_gexf(f"{self.DB_path}/database/{paper_index}/PSPP_{para_type}.gexf")

                # 노드 속성 합산
                for node_id, node_data in G.nodes(data=True):
                    if Total_G.has_node(node_id):
                        # 기존 노드가 있는 경우, 속성 값을 합산
                        existing_node = Total_G.nodes[node_id]
                        for attr_name, attr_value in node_data.items():
                            if not attr_name == "label":
                                existing_node[attr_name] = existing_node.get(attr_name, 0) + attr_value
                    else:
                        # 기존 노드가 없는 경우, 노드와 속성 추가
                        Total_G.add_node(node_id, **node_data)

                # 엣지 속성 합산
                for source, target, edge_data in G.edges(data=True):
                    if Total_G.has_edge(source, target):
                        # 기존 엣지가 있는 경우, 속성 값을 합산
                        existing_edge = Total_G[source][target]
                        for attr_name, attr_value in edge_data.items():
                            existing_edge[attr_name] = existing_edge.get(attr_name, 0) + attr_value
                    else:
                        # 기존 엣지가 없는 경우, 엣지와 속성 추가
                        Total_G.add_edge(source, target, **edge_data)

        # Reset Id of edges
        for i, edge in enumerate(Total_G.edges()):
            Total_G[edge[0]][edge[1]]['id'] = i

        # Add weight_rev for edges
        for edge in Total_G.edges():
            Total_G[edge[0]][edge[1]]['weight_rev'] = 1 / Total_G[edge[0]][edge[1]]['weight']

        nx.write_gexf(Total_G, f"{self.DB_path}/KARS/PSPP_{para_type}_network.gexf")