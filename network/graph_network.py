import os 
import shutil
import numpy as np
import networkx as nx
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import warnings
warnings.filterwarnings(action="ignore", category=FutureWarning)

class graph_network:
    def __init__(self, save_path, DB_name, text_type, modular_algorithm, filter_percent, community_seed, community_resolution):
        self.save_path = f"{save_path}/{DB_name}"
        self.DB_name = DB_name
        self.text_type = text_type
        self.modular_algorithm = modular_algorithm
        self.filter_percent = filter_percent
        self.community_seed = community_seed
        self.community_resolution = community_resolution
        # delete folder containing % sign
        for path, dirs, file in os.walk(self.save_path):
            for dir in dirs:
                if "%" in dir:
                    shutil.rmtree(f"{path}/{dir}")
        # load data
        with open(f"{self.save_path}/node_feature.json", "r", encoding="utf-8") as f:
            self.node_feature = json.load(f)
        with open(f"{self.save_path}/edge_feature.json", "r", encoding="utf-8") as f:
            self.edge_feature = json.load(f)   
        with open(f"{self.save_path}/{self.DB_name}_cleaned.jsonl", "r", encoding="utf-8") as f:
            self.metadata_list = [json.loads(line) for line in f]
        #for total
        self.graph_construct(freq="total")   
        self.pagerank(freq="total")
        self.keyword_filter(percent=self.filter_percent)
        self.research_structurization(freq="total")
        self.community_labeling()
        self.save_graph()
        self.save_subgraph()

    def graph_construct(self,freq):
        self.G = nx.Graph()
        node_list = []
        for key, value in self.node_feature.items():
            node_list.append((key,{"freq":value["year"][freq],"NER":value["NER"]}))
        self.G.add_nodes_from(node_list)

        edge_list = []
        for key, value in self.edge_feature.items():
            relation = key.split("-")
            edge_list.append((relation[0], relation[1], {"freq":value["year"][freq]}))
        self.G.add_edges_from(edge_list)

        self.total_nodes = self.G.number_of_nodes()
        print("Graph information")
        print("Node number: ", self.G.number_of_nodes())
        print("Edge number: ", self.G.number_of_edges())
     
    def pagerank(self, freq):
        node_pagerank = nx.pagerank(self.G, alpha=0.85, max_iter=20, tol=1e-06, weight=freq, dangling=None)
        for key, value in self.G.nodes.data():
            self.G.nodes[key]['pagerank'] = node_pagerank[key]
        print("Pagerank calculation finished")

    def keyword_filter(self, percent):
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
            if integrated_pagerank/total_pagerank*100 > percent:
                break
        remove_nodes = [node for node in self.G.nodes if node not in alive_nodes]

        # filter out nodes
        self.G_original = self.G.copy()
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
        plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
        # add horizontal line which y=self.total_pagerank and x=0~len(self.node_list)
        plt.plot([0,len(self.node_list)],[self.total_pagerank,self.total_pagerank], color='gray', linestyle='--', linewidth=0.5)
        # add vertical line which x=len(self.node_list) and y=0~self.total_pagerank
        plt.plot([len(self.node_list),len(self.node_list)],[0,self.total_pagerank], color='gray', linestyle='--', linewidth=0.5)
        # make 0,0 as origin
        plt.gca().set_xlim(left=0)
        plt.gca().set_ylim(bottom=0)
        plt.savefig(f"{self.save_path}/integrated_pagerank.png") 

    def research_structurization(self, freq):
        def modularity(self, G, freq):
            # calculate modularity using Louvain method
            if self.modular_algorithm == "louvain":
                node_modularity = nx.algorithms.community.louvain_communities(G, weight=freq, resolution=self.community_resolution, seed=self.community_seed)
            elif self.modular_algorithm == "greedy":
                node_modularity = nx.algorithms.community.greedy_modularity_communities(G, weight=freq, resolution=self.community_resolution)
            elif self.modular_algorithm == "girvan-newman":
                node_modularity = nx.algorithms.community.girvan_newman(G, weight=freq, resolution=self.community_resolution)
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
                node_modularity = modularity(self, G, freq=freq)
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
        # Save graph into gexf
        nx.write_gexf(self.G_original, f"{self.save_path}/graph_original.gexf")

        # Save graph into gexf
        nx.write_gexf(self.G, f"{self.save_path}/graph.gexf")

        # Save graph data as json
        with open(f"{self.save_path}/graph_original.json", "w") as f:
            json.dump(nx.node_link_data(self.G_original), f, indent=4)

        # Save graph data as json
        with open(f"{self.save_path}/graph.json", "w") as f:
            json.dump(nx.node_link_data(self.G), f, indent=4)


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
                if not os.path.exists(f"{self.save_path}/{folder_name}"):
                    os.makedirs(f"{self.save_path}/{folder_name}")

                # Save graph into gexf
                nx.write_gexf(subgraph, f"{self.save_path}/{folder_name}/graph.gexf")

                # Save graph data as json
                with open(f"{self.save_path}/{folder_name}/graph.json", "w") as f:
                    json.dump(nx.node_link_data(subgraph), f, indent=4)

                # Save node ranks as csv
                df = pd.DataFrame()
                df['node'] = [key for key, value in subgraph.nodes.data()]
                df['pagerank'] = [value['pagerank'] for key, value in subgraph.nodes.data()]
                df = df.sort_values(by=['pagerank'], ascending=False)
                df.to_csv(f"{self.save_path}/{folder_name}/pagerank.csv", index=False)

                # Save docs ranks as csv
                df = pd.DataFrame(columns=["conformity","title","DOI"])
                total_pagerank = sum([value['pagerank'] for key, value in subgraph.nodes.data()])
                for metadata in self.metadata_list:
                    if self.text_type == "abstract":
                        text_list = metadata['abstract_cleaned']
                        title = metadata['title']
                    elif self.text_type == "title":
                        text_list = metadata['title_cleaned']
                    doi = metadata['DOI']
                    conformity = 0
                    for node in subgraph.nodes:
                        for text in text_list:
                            if node in text:
                                conformity += subgraph.nodes[node]['pagerank']
                    if self.text_type == "abstract":
                        df = df.append({'DOI':doi, 'title':title, 'conformity':float(f"{conformity/total_pagerank*100:.2f}")}, ignore_index=True)
                    elif self.text_type == "title":
                        df = df.append({'DOI':doi, 'title':text_list, 'conformity':float(f"{conformity/total_pagerank*100:.2f}")}, ignore_index=True)
                df = df.sort_values(by=['conformity'], ascending=False)
                df.to_csv(f"{self.save_path}/{folder_name}/pagerank_doc.csv", index=False)

                # update progress bar
                pbar.update(1)

        print("Saving subgraph data is done")