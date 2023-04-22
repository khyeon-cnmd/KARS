import os 
import shutil
import re
import numpy as np
import networkx as nx
import jsonlines
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings(action="ignore", category=FutureWarning)

class research_structure:
    def __init__(self, save_path, DB_name, text_type, edge_type, start_year, end_year, modular_algorithm, modularity_resolution, modularity_seed, pagerank_filter, save_docs):
        self.save_path = save_path
        self.DB_name = DB_name
        self.text_type = text_type
        self.edge_type = edge_type
        self.start_year = start_year
        self.end_year = end_year
        self.modular_algorithm = modular_algorithm
        self.pagerank_filter = pagerank_filter
        self.community_seed = modularity_seed
        self.community_resolution = modularity_resolution
        self.save_docs = save_docs
        self.cv = CountVectorizer(ngram_range=(1,1), stop_words = None, lowercase=False, tokenizer=lambda x: x.split(' '))
        # delete folder containing % sign
        for path, dirs, file in os.walk(f"{self.save_path}/Research_structure"):
            for dir in dirs:
                if "%" in dir:
                    shutil.rmtree(f"{path}/{dir}")
        # load data
        self.metadata_list = list(jsonlines.open(f"{self.save_path}/Keyword_extraction/{self.DB_name}_cleaned.jsonl", "r"))
        # Structure the research field
        self.graph_dataficate()
        self.graph_construct()   
        self.pagerank()
        self.keyword_filter()
        self.research_structurization()
        self.community_labeling()
        self.save_graph()
        self.save_subgraph()

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

                # 2. year filtering
                year_range = (self.start_year, self.end_year)
                if not year in range(year_range[0], year_range[1]+1):
                    continue

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

        # save node & edge dict
        jsonlines.open(f"{self.save_path}/Research_structure/node_feature.json", mode='w').write(self.node_feature)
        jsonlines.open(f"{self.save_path}/Research_structure/edge_feature.json", mode='w').write(self.edge_feature)

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
     
    def pagerank(self):
        node_pagerank = nx.pagerank(self.G, alpha=0.85, max_iter=20, tol=1e-06, weight="total", dangling=None)
        for key, value in self.G.nodes.data():
            self.G.nodes[key]['pagerank'] = node_pagerank[key]
        self.G_original = self.G.copy()
        print("Pagerank calculation finished")

    def keyword_filter(self):
        # sort self.G.nodes by pagerank
        node_list = sorted(self.G.nodes, key=lambda x: self.G.nodes[x]['pagerank'], reverse=True)
        x = [i for i in range(0, len(node_list)+1)]
        y = [sum([self.G.nodes[node]['pagerank'] for node in node_list[:i]]) for i in range(1, len(node_list)+1)]
        y.insert(0,0)

        # filter in nodes until integrated pagerank is 80%
        integrated_pagerank = 0
        total_pagerank = float(sum([self.G.nodes[node]['pagerank'] for node in self.G.nodes]))
        alive_nodes = []
        for node in node_list:
            integrated_pagerank += self.G.nodes[node]['pagerank']
            alive_nodes.append(node)
            if integrated_pagerank/total_pagerank*100 > self.pagerank_filter:
                break
        remove_nodes = [node for node in self.G.nodes if node not in alive_nodes]

        # filter out nodes
        for node in remove_nodes:
            self.G.remove_node(node)

        # print graph information
        print("Keyword Filtered Graph information")
        print("Node number: ", self.G.number_of_nodes())
        print("Edge number: ", self.G.number_of_edges())

        # save integrated sum of pagerank using plt
        self.node_list = sorted(self.G.nodes, key=lambda x: self.G.nodes[x]['pagerank'], reverse=True)
        self.total_pagerank = sum([self.G.nodes[node]['pagerank'] for node in self.G.nodes])
        self.total_edge_weight = sum([self.G.edges[edge]['freq'] for edge in self.G.edges])
        plt.plot(x, y, color='red', linestyle='-', linewidth=1)
        plt.xlabel("Number of nodes")
        plt.ylabel("Integrated pagerank")
        plt.xticks([len(self.node_list), len(self.G_original.nodes)])
        plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
        # add horizontal line which y=self.total_pagerank and x=0~len(self.node_list)
        plt.plot([0,len(self.node_list)],[self.total_pagerank,self.total_pagerank], color='gray', linestyle='--', linewidth=0.5)
        # add vertical line which x=len(self.node_list) and y=0~self.total_pagerank
        plt.plot([len(self.node_list),len(self.node_list)],[0,self.total_pagerank], color='gray', linestyle='--', linewidth=0.5)
        # make 0,0 as origin
        plt.gca().set_xlim(left=0)
        plt.gca().set_ylim(bottom=0)
        plt.savefig(f"{self.save_path}/Research_structure/integrated_pagerank.png") 
        plt.close()

    def research_structurization(self):
        def modularity(self, G):
            # calculate modularity using Louvain method
            if self.modular_algorithm == "louvain":
                node_modularity = nx.algorithms.community.louvain_communities(G, weight="total", resolution=self.community_resolution, seed=self.community_seed)
            elif self.modular_algorithm == "greedy":
                node_modularity = nx.algorithms.community.greedy_modularity_communities(G, weight="total", resolution=self.community_resolution)
            elif self.modular_algorithm == "girvan-newman":
                node_modularity = nx.algorithms.community.girvan_newman(G, weight="total", resolution=self.community_resolution)
            # sorting community by size
            node_modularity = sorted(node_modularity, key=len, reverse=True)

            return node_modularity

        # 1. calculate freq sum of graph
        self.total_freq = sum([self.G.nodes[node]['freq'] for node in self.G.nodes])

        # 2. recursively modularize graph
        graphs = [self.G]
        modularized_graphs = []
        max_recursion = 1
        for i in range(0,max_recursion):
            queue_graphs = []
            for G in graphs:
                node_modularity = modularity(self, G)
                for idx, community in enumerate(node_modularity):
                    # make subgraph by community
                    subgraph = G.subgraph(community)
                    
                    # calculate pagerank sum of community
                    community_freq = sum([G.nodes[node]['freq'] for node in community])
                    print(f"community {idx} freq: {community_freq} ({community_freq/self.total_freq*100}%)")

                    # if pagerank sum of community is less than 10% of total pagerank, stop modularization and append graph to modularized_graphs
                    if community_freq/self.total_freq < 0.15:
                        modularized_graphs.append(subgraph)
                    # else, append subgraph to queue_graphs
                    else:
                        queue_graphs.append(subgraph)
            
            # print graphs and modularized_graphs
            print(f"unmodularized Graphs: {len(queue_graphs)}")
            print(f"modularized Graphs: {len(modularized_graphs)}")

            # if modularized_graphs is not empty, break loop
            graphs = queue_graphs
            if not graphs:
                break
        
        # unmodularized graphs are appended to modularized_graphs
        modularized_graphs.extend(graphs)

        # label community in self.G
        for idx, G in enumerate(modularized_graphs):
            for node in G.nodes:
                self.G.nodes[node]['community'] = idx

    def community_labeling(self):
        # 0. Assign colors
        colors = plt.cm.rainbow
       
        # 1. Calculate freq sum of each community
        community_freq = {}
        for node in self.G.nodes:
            community = self.G.nodes[node]['community']
            if community not in community_freq:
                community_freq[community] = 0
            community_freq[community] += self.G.nodes[node]['freq']

        # 2. Sort community_freq by freq
        community_list = sorted(community_freq, key=lambda x: community_freq[x], reverse=True)

        # 3. Get community nodes
        for idx, community in enumerate(community_list):
            community_nodes = [node for node in self.G.nodes if self.G.nodes[node]['community'] == community]
            community_nodes = sorted(community_nodes, key=lambda x: self.G.nodes[x]['pagerank'], reverse=True)
            community_freq = sum([self.G.nodes[node]['freq'] for node in community_nodes])
        
            #print top 30 nodes in community
            print(f"Community {community}, node number: {len(community_nodes)}, freq: {community_freq/self.total_freq*100:.2f}%")
            print("==============================")
            label = ""
            for idx2, node in enumerate(community_nodes):
                print(f"{idx2+1} {node} {self.G.nodes[node]['pagerank']}")
                label = label + node + " "
                if idx2 == 4:
                    break

            #get label of community by input
            #label = input("Write the label of community (Enter=unrelated): ")
            #if label == "":
            #    label = "unrelated"
            
            # assign label to community
            for node in community_nodes:
                self.G.nodes[node]['community'] = label
                self.G.nodes[node]['color'] = matplotlib.colors.to_hex(colors(idx/len(community_list)))

    def save_graph(self):
        # Save original graph
        nx.write_gexf(self.G_original, f"{self.save_path}/Research_structure/graph_original.gexf")
        jsonlines.open(f"{self.save_path}/Research_structure/graph_original.json", mode='w').write(nx.node_link_data(self.G_original))

        # Save graph into gexf
        nx.write_gexf(self.G, f"{self.save_path}/Research_structure/graph.gexf")
        jsonlines.open(f"{self.save_path}/Research_structure/graph.json", mode='w').write(nx.node_link_data(self.G))


        print("Saving graph data is done")

    def save_subgraph(self):
        # get community labels from self.G
        community_list = set([self.G.nodes[node]['community'] for node in self.G.nodes])

        # sort community by pagerank
        community_freq = {}
        for node in self.G.nodes:
            community = self.G.nodes[node]['community']
            if community not in community_freq:
                community_freq[community] = 0
            community_freq[community] += self.G.nodes[node]['freq']

        community_list = sorted(community_freq, key=lambda x: community_freq[x], reverse=True)

        with tqdm(total=len(community_list), desc="Save subgraph") as pbar:
            for idx, community in enumerate(community_list):
                # find nodes in community
                community_nodes = [node for node in self.G.nodes if self.G.nodes[node]['community'] == community]

                # make subgraph by community
                subgraph = self.G.subgraph(community_nodes)

                # make folder for each community
                subgraph_freq = sum([subgraph.nodes[node]['freq'] for node in subgraph.nodes])
                percent_of_community = round(subgraph_freq/self.total_freq*100, 2)
                folder_name = f"{idx} {community} ({percent_of_community}%)" 
                print(f"Community '{community}' has {len(community_nodes)} nodes, {percent_of_community}% of total nodes")
                if not os.path.exists(f"{self.save_path}/Research_structure/{folder_name}"):
                    os.makedirs(f"{self.save_path}/Research_structure/{folder_name}")

                # Save graph into gexf
                nx.write_gexf(subgraph, f"{self.save_path}/Research_structure/{folder_name}/graph.gexf")

                # Save graph data as json
                jsonlines.open(f"{self.save_path}/Research_structure/{folder_name}/graph.json", mode='w').write(nx.node_link_data(subgraph))

                # Save node ranks as csv
                df = pd.DataFrame()
                df['node'] = [key for key, value in subgraph.nodes.data()]
                df['pagerank'] = [value['pagerank'] for key, value in subgraph.nodes.data()]
                df = df.sort_values(by=['pagerank'], ascending=False)
                df.to_csv(f"{self.save_path}/Research_structure/{folder_name}/pagerank.csv", index=False)

                # Save docs ranks as csv
                if self.save_docs == True:
                    df = pd.DataFrame(columns=["conformity","title","DOI"])
                    total_pagerank = sum([value['pagerank'] for key, value in subgraph.nodes.data()])
                    for metadata in self.metadata_list:
                        text_list = metadata['title_cleaned']
                        doi = metadata['DOI']
                        conformity = 0
                        for node in subgraph.nodes:
                            for text in text_list:
                                if node in text:
                                    conformity += subgraph.nodes[node]['pagerank']
                        df = df.append({'DOI':doi, 'title':text_list[0], 'conformity':float(f"{conformity/total_pagerank*100:.2f}")}, ignore_index=True)
                    df = df.sort_values(by=['conformity'], ascending=False)
                    df.to_csv(f"{self.save_path}/Research_structure/{folder_name}/pagerank_doc.csv", index=False)

                # update progress bar
                pbar.update(1)

        print("Saving subgraph data is done")