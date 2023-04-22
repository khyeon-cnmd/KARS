import os 
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import shutil
import jsonlines
import networkx as nx
import pandas as pd
import json
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from sklearn.feature_extraction.text import CountVectorizer
warnings.filterwarnings(action="ignore", category=FutureWarning)

class relationship_analysis:
    def __init__(self, save_path, DB_name):
        self.save_path = save_path
        self.DB_name = DB_name
        self.scan_keyword_list()

    def __call__(self, material, text_type, edge_type):
        self.material = material
        self.text_type = text_type
        self.edge_type = edge_type
        self.cv = CountVectorizer(ngram_range=(1,1), stop_words = None, lowercase=False, tokenizer=lambda x: x.split(' '))
        self.material_related_paper()
        self.graph_dataficate()
        self.graph_construct()

    def scan_keyword_list(self):
        # 1. Load graph network
        self.G_original = nx.read_gexf(f"{self.save_path}/Research_structure/graph_original.gexf")
        self.G_filter = nx.read_gexf(f"{self.save_path}/Research_structure/graph.gexf")
        
        # 2. Get materials list and make df
        material_dict = {}
        for node in self.G_original.nodes:
            if self.G_original.nodes[node]['NER'] == 'material':
                material_dict[node] = self.G_original.nodes[node]['pagerank']
        # make df
        material_df = pd.DataFrame()
        # make list sorted by pagerank
        material_df["material"] = material_dict.keys()
        material_df["pagerank"] = material_dict.values()
        material_df = material_df.sort_values(by="pagerank", ascending=False)
        material_df["material"] = material_df["material"] + "(" + material_df["pagerank"].astype(str) + ")"
        material_df.drop("pagerank", axis=1, inplace=True)


        # 3. Get value list
        value_dict = {}
        for node in self.G_original.nodes:
            if self.G_original.nodes[node]['NER'] == 'value':
                value_dict[node] = self.G_original.nodes[node]['pagerank']
        # make df
        value_df = pd.DataFrame()
        # make list sorted by pagerank
        value_df["value"] = value_dict.keys()
        value_df["pagerank"] = value_dict.values()
        value_df = value_df.sort_values(by="pagerank", ascending=False)
        value_df["value"] = value_df["value"] + "(" + value_df["pagerank"].astype(str) + ")"        
        value_df = value_df.drop("pagerank", axis=1)

        # 4. Get keyword list
        df = pd.DataFrame()
        for root, dirs, files in os.walk(f"{self.save_path}/Research_structure"):
            for dir in dirs:
                pagerank = pd.read_csv(f"{self.save_path}/Research_structure/{dir}/pagerank.csv")
                df[dir] = pagerank["node"] + "(" + pagerank["pagerank"].astype(str) + ")"
            
        # 5. merge value and material df onto df reset index
        df = pd.concat([material_df, value_df, df], axis=1, ignore_index=True)

        # 7. Save df
        df.to_csv(f"{self.save_path}/Relationship_analysis/keyword_list.csv")
        
    def material_related_paper(self):
        all_metadata_list = []
        # 1. Get metadata
        with jsonlines.open(f"{self.save_path}/Keyword_extraction/{self.DB_name}_cleaned.jsonl", 'r') as f:
            for line in f.iter():
                all_metadata_list.append(line)
        print(f"Number of papers: {len(all_metadata_list)}")

        # 2. filter metadata by material
        self.metadata_list = []
        for metadata in all_metadata_list:
            if self.text_type == "title" and "title_cleaned" in metadata.keys():
                text_list = metadata["title_cleaned"]
            elif self.text_type == "abstract" and "abstract_cleaned" in metadata.keys():
                text_list = metadata["abstract_cleaned"]              
            else:
                continue
            for text in text_list:
                if self.material in text:
                    self.metadata_list.append(metadata)
                    break
        print(f"Number of {self.material} related papers: {len(self.metadata_list)}")

    def graph_dataficate(self):
        def NER(word):
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

            # Find value
            # if first letter is number
            if word[0].isdigit():
                return "value"

            # Find device
            if "/" in word:
                for w in word.split("/"):
                    if count_upper(word) < round(len(word)/2):
                        continue
                    if count_upper(word) == count_element(word):
                        return "device"

            # Find material
            if count_upper(word) >= round(len(word)/2) and count_upper(word) == count_element(word):
                return "material"

            # Rest are other
            return "other"

        def node_extraction(text_list):
            for text in text_list:
                keyword_list = text.split(" ")
                for keyword in keyword_list:
                    if not keyword == "":
                        # 1. year freq feature
                        if not keyword in self.node_feature.keys():
                            self.node_feature[keyword] = {"year":{"total":0}, "NER":None}
                        if not year in self.node_feature[keyword]["year"].keys():
                            self.node_feature[keyword]["year"][year] = 0

                        self.node_feature[keyword]["year"]["total"] += 1
                        self.node_feature[keyword]["year"][year] += 1

                        # 2. NER feature
                        self.node_feature[keyword]["NER"] = NER(keyword)

        def neighbor_edge_extraction(text_list):
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
                                    if not edge_name in self.edge_feature.keys():
                                        self.edge_feature[edge_name] = {"year":{"total":0}}
                                    if not year in self.edge_feature[edge_name]["year"].keys():
                                        self.edge_feature[edge_name]["year"][year] = 0
                                    self.edge_feature[edge_name]["year"]["total"] += 1
                                    self.edge_feature[edge_name]["year"][year] += 1

        def co_occurrence_edge_extraction(text_list):
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
                                if not edge_name in self.edge_feature.keys():
                                    self.edge_feature[edge_name] = {"year":{"total":0}}
                                if not year in self.edge_feature[edge_name]["year"].keys():
                                    self.edge_feature[edge_name]["year"][year] = 0
                                self.edge_feature[edge_name]["year"]["total"] += count
                                self.edge_feature[edge_name]["year"][year] += count

        print("Node & Edge extraction...")
        self.node_feature = {}
        self.edge_feature = {}
        with tqdm(total=len(self.metadata_list)) as pbar:
            for metadata in self.metadata_list:
                # 1. Get year
                if 'published-print' in metadata.keys():
                    year = metadata["published-print"]["date-parts"][0][0]
                elif 'published-online' in metadata.keys():
                    year = metadata["published-online"]["date-parts"][0][0]

                # 3. Get text
                if self.text_type == "title" and "title_cleaned" in metadata.keys():
                    text_list = metadata["title_cleaned"]
                elif self.text_type == "abstract" and "abstract_cleaned" in metadata.keys():
                    text_list = metadata["abstract_cleaned"]              
                else:
                    continue

                # 4. Node extraction
                node_extraction(text_list)
            
                # 5 Edge extraction
                if self.edge_type == "neighbor":
                    neighbor_edge_extraction(text_list)

                elif self.edge_type == "co-occurrence":
                    co_occurrence_edge_extraction(text_list)

                pbar.update(1)

    def graph_construct(self):
        self.G = nx.Graph()
        node_list = []
        for key, value in self.node_feature.items():
            node_list.append((key,{"freq":value["year"]["total"],"NER":value["NER"]}))
        self.G.add_nodes_from(node_list)

        edge_list = []
        for key, value in self.edge_feature.items():
            relation = key.split("-")
            edge_list.append((relation[0], relation[1], {"freq":value["year"]["total"]}))
        self.G.add_edges_from(edge_list)

        self.total_nodes = self.G.number_of_nodes()
        print("Graph information")
        print("Node number: ", self.G.number_of_nodes())
        print("Edge number: ", self.G.number_of_edges())

        # return info of graph
        self.info = {"node_number":self.G.number_of_nodes(), "edge_number":self.G.number_of_edges()}

    def path_find(self, source, target, internodes, NER_list, max_distance):
        path_length = {}
        for path in tqdm(nx.all_simple_paths(self.G, source=source, target=target, cutoff=max_distance)):
            # filter by internodes
            if not len(internodes) == 0:
                if not set(internodes).issubset(set(path)):
                    continue

            # filter NER options
            # make set of NER in path
            NER_set = set()
            for node in path[1:-1]:
                NER_set.add(self.G.nodes[node]['NER'])
            # if NER_list is subset of NER_set, continue
            if not set(NER_list).issubset(NER_set):
                continue
                
            path_name = path[0]

            # get minimum edge freq of edges in path
            prev_edge_freq = 1000000
            edge_freq = 0
            for i in range(len(path)-1):
                edge_freq = self.G.edges[path[i],path[i+1]]['freq']
                if edge_freq < prev_edge_freq:
                    prev_edge_freq = edge_freq
                
                path_name += "-(" + str(edge_freq) + ")-" + path[i+1]


            # get path length
            path_length[path_name] = (prev_edge_freq, len(path)-1)

        # sort by edge_freq
        if not len(path_length) == 0:
            path_length = sorted(path_length.items(), key=lambda x: x[1][0], reverse=True)
            largest_freq = path_length[0][1][0]
            path_length = [path for path in path_length if path[1][0] == largest_freq]
            
            # sort by cutoff 
            path_length = sorted(path_length, key=lambda x: x[1][1], reverse=False)
        else:
            path_length = []

        return path_length

class relationship_analysis_old:
    def __init__(self, save_path, ):
        self.save_path = save_path
        self.DB_name = DB_name
        self.text_type = text_type
        self.scan_materials_list()
        self.material = input("What materials do you want to analyze?: ")
        if not os.path.exists(f"{self.save_path}/{self.DB_name}/{self.material}_graph.gexf"):
            self.material_related_papers()
            self.node_dict = {}
            self.edge_dict = {}
            self.graph_construct()
        else:
            self.G = nx.read_gexf(f"{self.save_path}/{self.DB_name}/{self.material}_graph.gexf")


    def rule_based_NER(self):
        # 0. Assign colors of 1) material 2) processing 3) structure 4) property 5) performance Enter) others
        color_dict = {"material":"#36C5F0", "processing":"#E01E5A", "structure":"ECB22E", "property":"2EB67D","performance":"4A154B","others":"#737373"}
       
        # 1. Calculate pagerank of each community
        community_pagerank = {}
        for node in self.G.nodes:
            community = self.G.nodes[node]['community']
            if community not in community_pagerank:
                community_pagerank[community] = 0
            community_pagerank[community] += self.G.nodes[node]['pagerank']

        # 2. Sort community by pagerank
        community_list = sorted(community_pagerank, key=lambda x: community_pagerank[x], reverse=True)
        print(community_list)

        # 3. Get community nodes
        for idx, community in enumerate(community_list):
            community_nodes = [node for node in self.G.nodes if self.G.nodes[node]['community'] == community]
            community_nodes = sorted(community_nodes, key=lambda x: self.G.nodes[x]['pagerank'], reverse=True)
            community_pagerank = sum([self.G.nodes[node]['pagerank'] for node in community_nodes])
        
            #print top 30 nodes in community
            print(f"Community {community}, node number: {len(community_nodes)}, pagerank: {community_pagerank/self.total_pagerank*100:.2f}%")
            print("==============================")
            idx2=0
            for node in community_nodes:
                if self.G.nodes[node]['NER'] == 'material' or self.G.nodes[node]['NER'] == 'device':
                    continue
                else:
                    print(f"{idx2+1} {node} {self.G.nodes[node]['pagerank']}")
                    idx2+=1
                if idx2 == 30:
                    break

            #get label of community by input
            label = input("Choose community label. 1) material 2) processing 3) structure 4) property 5) performance Enter) others\n:")
            if label == '1':
                label = 'material'
            elif label == '2':
                label = 'processing'
            elif label == '3':
                label = 'structure'
            elif label == '4':
                label = 'property'
            elif label == '5':
                label = 'performance'
            else:
                label = 'others'
            
            # assign label to NER
            for node in community_nodes:
                if self.G.nodes[node]['NER'] == 'material' or self.G.nodes[node]['NER'] == 'device':
                    self.G.nodes[node]["color"] = color_dict[label]
                else:
                    self.G.nodes[node]['NER'] = label
                    self.G.nodes[node]['color'] = color_dict[label]
        
    def NER_to_csv(self):
        # make pandas dataframe from NER result
        self.NER_df = pd.DataFrame(columns=['material', 'material_PR', 'device', 'device_PR', 'process', 'process_PR' 'structure', 'structure_PR', 'property', 'property_PR','performance', 'performance_PR'])
        for NER in ['material', 'device', 'process', 'structure', 'property','performance']:
            G = self.G.subgraph([node for node in self.G.nodes if self.G.nodes[node]['NER'] == NER])
            node_list = sorted(G.nodes, key=lambda x: G.nodes[x]['pagerank'], reverse=True)
            for i, node in enumerate(node_list):
                self.NER_df.loc[i, NER] = node
                self.NER_df.loc[i, f"{NER}_PR"] = G.nodes[node]['pagerank']

        # save NER_df to csv
        self.NER_df.to_csv(f"{self.save_path}/MPSPP_df.csv", index=False)

    def MPSPP_graph_query(self):
        nodes_dict = {"material":[], "device":[], "process":[], "structure":[], "property":[], "performance":[]}
        while 1:
            material = input("Material's name (Enter=exit): ")
            if material == "":
                break
            else:
                nodes_dict["material"].append(material)
        if len(nodes_dict["material"]) == 0:
            nodes_dict["material"] = [node for node in self.G.nodes if self.G.nodes[node]['NER'] == "material"]
        
        #while 1:
        #    device = input("Device's name (Enter=exit): ")
        #    if device == "":
        #        break
        #    else:
        #        nodes_dict["device"].append(device)
        #if len(nodes_dict["device"]) == 0:
        #    nodes_dict["device"] = [node for node in self.G.nodes if self.G.nodes[node]['NER'] == "device"]

        while 1:
            process = input("Process's name (Enter=exit): ")
            if process == "":
                break
            else:
                nodes_dict["process"].append(process)
        if len(nodes_dict["process"]) == 0:
            nodes_dict["process"] = [node for node in self.G.nodes if self.G.nodes[node]['NER'] == "process"]

        while 1:
            structure = input("Structure's name (Enter=exit): ")
            if structure == "":
                break
            else:
                nodes_dict["structure"].append(structure)
        if len(nodes_dict["structure"]) == 0:
            nodes_dict["structure"] = [node for node in self.G.nodes if self.G.nodes[node]['NER'] == "structure"]

        while 1:
            property = input("Property's name (Enter=exit): ")
            if property == "":
                break
            else:
                nodes_dict["property"].append(property)
        if len(nodes_dict["property"]) == 0:
            nodes_dict["property"] = [node for node in self.G.nodes if self.G.nodes[node]['NER'] == "property"]

        while 1:
            performance = input("Performance's name (Enter=exit): ")
            if performance == "":
                break
            else:
                nodes_dict["performance"].append(performance)
        if len(nodes_dict["performance"]) == 0:
            nodes_dict["performance"] = [node for node in self.G.nodes if self.G.nodes[node]['NER'] == "performance"]

        # make paths
        paths = []
        for material in nodes_dict["material"]:
            for process in nodes_dict["process"]:
                for structure in nodes_dict["structure"]:
                    for property in nodes_dict["property"]:
                        for performance in nodes_dict["performance"]:
                            paths.append([material, process, structure, property, performance])

        # calculate path weight
        path_weights = {}
        for path in paths:
            path_weights[tuple(path)] = nx.path_weight(self.G, path, weight='freq')

        # sort path by weight
        sorted_path_weights = sorted(path_weights.items(), key=lambda x: x[1], reverse=True)

        # save path to csv
        path_df = pd.DataFrame(columns=['material', 'process', 'structure', 'property', 'performance', 'weight'])
        for i, path in enumerate(sorted_path_weights):
            path_df.loc[i, 'material'] = path[0][0]
            path_df.loc[i, 'process'] = path[0][1]
            path_df.loc[i, 'structure'] = path[0][2]
            path_df.loc[i, 'property'] = path[0][3]
            path_df.loc[i, 'performance'] = path[0][4]
            path_df.loc[i, 'weight'] = path[1]
        path_df.to_csv(f"{self.save_path}/MPSPP_path_df.csv", index=True)
