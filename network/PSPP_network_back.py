import os
import sys
import json
import pandas as pd
import spacy 
import networkx as nx
import scispacy
from spacy.tokenizer import Tokenizer
import re
from tqdm import tqdm
from pymongo import MongoClient

class PSPP_network:
    def __init__(self, DB_path, tokenized_dict, entity_dict, edge_type):
        self.DB_path = DB_path + "/KARS"
        if os.path.isdir(self.DB_path) == False:
            os.mkdir(self.DB_path)
        self.tokenized_dict = tokenized_dict
        self.entity_dict = entity_dict
        self.edge_type = edge_type
        self.spacy_model = spacy.load("en_core_sci_sm")
        self.spacy_model.tokenizer = Tokenizer(self.spacy_model.vocab, token_match=re.compile(r'\S+').match)
    
    def construct_PSPP_network(self):
        print("Constructing PSPP network...")
        # 1. Select graph type by edge_type
        if self.edge_type == "co-occurrence":
            self.G = nx.Graph()
        elif self.edge_type == "neighbor":
            self.G = nx.DiGraph()

        # 2. add nodes and edges
        for paper_index in self.tokenized_dict.keys():
            for para_index in self.tokenized_dict[paper_index].keys():
                if para_index == "year":
                    continue
                year = self.tokenized_dict[paper_index]["year"][0]
                if not len(str(year)) == 4:
                    continue

                for sen_index in self.tokenized_dict[paper_index][para_index].keys():
                    if self.edge_type == "co-occurrence": # 문장 단위로 모든 개체들의 공존 빈도를 계산
                        entity_full_string = ""
                        for entity, count in self.entity_dict[paper_index][para_index][sen_index].items():
                            for i in range(count):
                                entity_full_string += entity + " "
                        
                        entity_full_string = entity_full_string.strip()
                        entity_list = entity_full_string.split(" ")
                        for i in range(len(entity_list)):
                            for j in range(i+1, len(entity_list)):
                                if not self.G.has_edge(entity_list[i], entity_list[j]):
                                    self.G.add_edge(entity_list[i], entity_list[j], weight=1)
                                else:
                                    self.G.edges[entity_list[i], entity_list[j]]["weight"] += 1

                                # node 빈도 추가
                                if entity_list[i] not in self.G.nodes():
                                    self.G.add_node(entity_list[i], weight=1)
                                else:
                                    self.G.nodes[entity_list[i]]["weight"] += 1

                                # year 정보 추가
                                if str(year) not in self.G.nodes[entity_list[i]]:
                                    self.G.nodes[entity_list[i]][str(year)] = 1
                                else:
                                    self.G.nodes[entity_list[i]][str(year)] += 1

                    elif self.edge_type == "neighbor": # 문장 단위로 개체들의 좌->우 인접 빈도를 계산
                        entity_list = list(self.entity_dict[paper_index][para_index][sen_index].keys())
                        root_word_list = [entity.split(" ")[-1] for entity in entity_list]

                        # 1. parent root 연결
                        root_connected_token = []
                        for child_token_lemma_pos_dep in self.tokenized_dict[paper_index][para_index][sen_index]:
                            child_token, child_lemma, child_pos, child_dep = child_token_lemma_pos_dep
                            if child_lemma in root_word_list:
                                token = child_token
                                while True:
                                    parent_token = token.head
                                    parent_token = [token for token in self.tokenized_dict[paper_index][para_index][sen_index] if token[0] == parent_token][0]
                                    parent_token, parent_lemma, parent_pos, parent_dep = parent_token

                                    if parent_lemma in root_word_list:
                                        # edge 추가
                                        if not self.G.has_edge(child_lemma, parent_lemma):
                                            self.G.add_edge(child_lemma, parent_lemma, weight=1)
                                        else:
                                            self.G.edges[child_lemma, parent_lemma]["weight"] += 1
                                        break
                                    elif parent_dep == "ROOT":
                                        root_connected_token.append(child_token)
                                        break
                                    token = parent_token

                        # root 에 연결된 token에 대해서 의존관계 재정립
                        for root_token in root_connected_token:
                            if root_token.dep_ == "nsubjpass" or root_token.dep_ == "dobj": # 수동태 주어 우선 -> 목적어 우선
                                new_root = [token for token in self.tokenized_dict[paper_index][para_index][sen_index] if token[0] == root_token][0]
                                break

                        # 새로운 root 에 연결된 token에 대해서 의존관계 재정립
                        for other_token in root_connected_token:
                            if not other_token == new_root[0]:
                                other_token = [token for token in self.tokenized_dict[paper_index][para_index][sen_index] if token[0] == other_token][0]
                                if not self.G.has_edge(new_root[1], other_token[1]):
                                    self.G.add_edge(new_root[1], other_token[1], weight=1)
                                else:
                                    self.G.edges[new_root[1], other_token[1]]["weight"] += 1

                        # node 빈도 추가
                        for root_word in root_word_list:
                            if not root_word in self.G.nodes():
                                self.G.add_node(root_word, weight=1)

                            if not "weight" in self.G.nodes[root_word]:
                                self.G.nodes[root_word]["weight"] = 1
                            else:
                                self.G.nodes[root_word]["weight"] += 1

                            # year 정보 추가
                            if str(year) not in self.G.nodes[root_word]:
                                self.G.nodes[root_word][str(year)] = 1
                            else:
                                self.G.nodes[root_word][str(year)] += 1
  
        # 3. Add weight reverse edge
        for edge in self.G.edges():
            weight_rev = 1 / self.G.edges[edge]["weight"]
            self.G.edges[edge]["weight_rev"] = weight_rev

        # 3. save graph
        print("save graph")
        nx.write_gexf(self.G, os.path.join(self.DB_path, "PSPP_network.gexf"))

        return self.G

    def construct_PSPP_tree(self, count_limit):
        print("construct_PSPP_tree")
        # 1. make tree graph
        self.tree_G = nx.DiGraph()

        # 2. get self.G's root nodes as list
        self.G_copy = self.G.copy()

        # 3. filter root nodes
        for root_node in self.G.nodes():
            # 길이가 1글자면 제외
            if len(root_node) <= 1:
                self.G_copy.remove_node(root_node)

            # 알파벳이 아니면 제외
            # elif re.sub(r"[^a-zA-Z]", "", root_node) == "":
            #     self.G_copy.remove_node(root_node)

            # 빈도수가 1이면 제외
            elif self.G.nodes[root_node]["weight"] <= count_limit:
                self.G_copy.remove_node(root_node)

        # 4. find root nodes 
        depth = 0
        found_targets = [] 
        for node in tqdm(self.G_copy.copy().nodes()):
            root_node = False
            for node2 in self.G_copy.copy().nodes():
                if node == node2:
                    continue

                if not self.G.has_edge(node, node2):
                    node_to_node2_weight = 0
                else:
                    node_to_node2_weight = self.G.edges[node, node2]["weight"]

                if not self.G.has_edge(node2, node):
                    node2_to_node_weight = 0
                else:
                    node2_to_node_weight = self.G.edges[node2, node]["weight"]

                if node_to_node2_weight == 0 and node2_to_node_weight == 0:
                    continue

                # root_node 가 모든 root_node2 대비 좌->우 인접 빈도가 더 크고 root_node 의 빈도가 root_node2 의 빈도보다 크면 진짜 root node
                if node_to_node2_weight > node2_to_node_weight and self.G.nodes[node]["weight"] < self.G.nodes[node2]["weight"]:
                    root_node = False
                    break
                else:
                    root_node = True

            if root_node == True:
                self.tree_G.add_node(node, depth=depth)
                found_targets.append(node)
                
        print("depth: ", depth, "root_nodes: ", self.tree_G.nodes())

        # 5. Find child nodes using shortest path
        depth = -1
        while len(found_targets) > 0:
            total_found_sources = []
            for target in tqdm(found_targets):
                path_dict = {}
                found_sources = []

                # 1. Find shortest paths for target
                for source in self.G.nodes():
                    if not source == target:
                        try:
                            path = nx.shortest_path(self.G_copy, source=source, target=target, weight='weight_rev')
                            path_dict[source] = path
                        except:
                            pass
                
                # 2. find nodes having path length  = 2
                for source, path_list in path_dict.items():
                    if len(path_list) == 2:
                        found_sources.append(source)

                # 3. labeling and add to subgraph
                for node in found_sources:
                    self.tree_G.add_edge(node, target, weight=self.G_copy[node][target]["weight"])
                    if "depth" not in self.tree_G.nodes[node].keys():
                        self.tree_G.add_node(node, weight=self.G_copy.nodes[node]["weight"], depth=depth)

                # 4. add found_nodes to total_found_nodes
                total_found_sources += found_sources

            # 5. remove found_targets from G
            self.G_copy.remove_nodes_from(found_targets)

            # 6. set total_found_nodes to new found_targets
            found_targets = list(set(total_found_sources))
            
            print(f"depth: {depth} / found_targets: {found_targets}")

            # 7. increase depth
            depth += -1
    
        # save tree graph
        print("save tree graph")
        nx.write_gexf(self.tree_G, os.path.join(self.DB_path, "PSPP_tree.gexf"))

        return self.tree_G

          



