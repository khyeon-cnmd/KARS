import os 
import shutil
import numpy as np
import networkx as nx
import json
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings(action="ignore", category=FutureWarning)

class MPSPP_analysis:
    def __init__(self, save_path, DB_name):
        self.save_path = save_path
        self.DB_name = DB_name
        self.G = nx.read_gexf(f"{save_path}/{DB_name}/graph.gexf")
        #self.NER_to_csv()
        #self.MPSPP_graph_query()

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