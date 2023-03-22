import os 
import shutil
import networkx as nx
import json
import pandas as pd
from matplotlib.pyplot import figure, text
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings(action="ignore", category=FutureWarning)

class graph_network:
    def __init__(self, save_path, DB_name, filter_percent, community_seed, community_resolution):
        self.save_path = f"{save_path}/{DB_name}"
        self.DB_name = DB_name
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
        #self.synonyms() 
        self.modularity(freq="total")
        self.node_size()
        self.edge_width()
        self.save_subgraph()
        # Save graph data as json
        with open(f"{self.save_path}/graph.json", "w") as f:
            json.dump(nx.node_link_data(self.G), f, indent=4)
        
    def __call__(self, save_path):
        self.save_path = save_path
        with open(f"{save_path}/graph.json", "r", encoding="utf-8") as f:
            subgraph = json.load(f)
        self.G = nx.node_link_graph(subgraph)
        self.modularity(freq="total")
        self.save_subgraph()
        # Save graph data as json
        with open(f"{self.save_path}/graph.json", "w") as f:
            json.dump(nx.node_link_data(self.G), f, indent=4)

    def graph_construct(self,freq):
        self.G = nx.Graph()
        node_list = []
        for key, value in self.node_feature.items():
            node_list.append((key,{"freq":value[freq],"pagerank":None,"modularity":None}))
        self.G.add_nodes_from(node_list)

        edge_list = []
        for key, value in self.edge_feature.items():
            relation = key.split("-")
            edge_list.append((relation[0], relation[1], {"freq":value[freq]}))
        self.G.add_edges_from(edge_list)

        self.total_nodes = self.G.number_of_nodes()
        print("Graph information")
        print("Node number: ", self.G.number_of_nodes())
        print("Edge number: ", self.G.number_of_edges())

    def synonyms(self):
        # finding synonyms using edge linkage information
        # if two nodes are connected by more than 50% of total edges, they are synonyms

        # sort self.G.nodes by freq
        node_list = sorted(self.G.nodes, key=lambda x: self.G.nodes[x]['freq'], reverse=True)

        # find synonyms using SimRank similarity
        # SimRank similarity is a measure of similarity between two nodes in a graph
        for node in node_list:
            simrank = nx.simrank_similarity(self.G, source=node, target=None, importance_factor=0.1, max_iterations=1000, tolerance=0.0001)
            print(node, sorted(simrank.items(), key=lambda x: x[1], reverse=True)[:10])            
            #panther = nx.panther_similarity(self.G, source=node, k=5, path_length=5, c=0.5, delta=0.1, eps=None)
            #print(node, sorted(panther.items(), key=lambda x: x[1], reverse=True)[:10])

    def pagerank(self, freq):
        node_pagerank = nx.pagerank(self.G, alpha=0.85, max_iter=20, tol=1e-06, weight=freq, dangling=None)
        for key, value in self.G.nodes.data():
            self.G.nodes[key]['pagerank'] = node_pagerank[key]

    def keyword_filter(self, percent):
        # sort self.G.nodes by pagerank
        node_list = sorted(self.G.nodes, key=lambda x: self.G.nodes[x]['pagerank'], reverse=True)
        x = [i for i in range(0, len(node_list)+1)]
        y = [sum([self.G.nodes[node]['pagerank'] for node in node_list[:i]]) for i in range(1, len(node_list)+1)]
        y.insert(0,0)

        # filter in nodes until integrated pagerank is 80%
        integrated_pagerank = 0
        total_pagerank = sum([self.G.nodes[node]['pagerank'] for node in self.G.nodes])
        alive_nodes = []
        for node in node_list:
            integrated_pagerank += self.G.nodes[node]['pagerank']
            alive_nodes.append(node)
            if integrated_pagerank/total_pagerank*100 > percent:
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

    def modularity(self, freq):
        self.node_modularity = nx.algorithms.community.louvain_communities(self.G, weight=freq, resolution=self.community_resolution, seed=self.community_seed)
        # sorting community by size
        self.node_modularity = sorted(self.node_modularity, key=len, reverse=True)
        for key, value in self.G.nodes.data():
            for idx, community in enumerate(self.node_modularity):
                if key in community:
                    self.G.nodes[key]['modularity'] = idx
                    break
        
    def node_size(self):
        def min_max_normalize(x, min, max):
            return (x-min)/(max-min)
        
        max_pagerank = max([value['pagerank'] for key, value in self.G.nodes.data()])
        min_pagerank = min([value['pagerank'] for key, value in self.G.nodes.data()])
        for key, value in self.G.nodes.data():
            self.G.nodes[key]['node_size'] = float(min_max_normalize(value['pagerank'], min_pagerank, max_pagerank)*10000)

    def edge_width(self):
        def min_max_normalize(x, min, max):
            return (x-min)/(max-min)
        
        max_freq = max([value['freq'] for key1, key2, value in self.G.edges.data()])
        min_freq = min([value['freq'] for key1, key2, value in self.G.edges.data()])
        for key1, key2, value in self.G.edges.data():
            self.G.edges[(key1,key2)]['edge_width'] = float(min_max_normalize(value['freq'], min_freq, max_freq)*10)

    def save_subgraph(self):
        # Set overall figure size
        plt.figure(figsize=(20,20), dpi=300).tight_layout()

        with tqdm(total=len(self.node_modularity), desc="Save graph") as pbar:
            for community in self.node_modularity:
                # make subgraph by community
                def filter_node(n1):
                    return n1 in community
                subgraph = nx.subgraph_view(self.G, filter_node=filter_node)

                # make folder for each community
                community_index = self.node_modularity.index(community)
                #subgraph_freq = sum([subgraph.nodes[node]['freq'] for node in subgraph.nodes])
                subgraph_pagerank = sum([subgraph.nodes[node]['pagerank'] for node in subgraph.nodes])
                percent_of_community = round(subgraph_pagerank/self.total_pagerank*100, 2)
                folder_name = f"{community_index} ({percent_of_community}%)" 
                print(f"Community {community_index} has {len(community)} nodes, {percent_of_community}% of total nodes")
                if not os.path.exists(f"{self.save_path}/{folder_name}"):
                    os.makedirs(f"{self.save_path}/{folder_name}")
                
                # Set layout and position of nodes. k is distance btw nodes
                pos = nx.spring_layout(G=subgraph, k=5, weight='freq', iterations=100, scale=1, center=None, dim=2, seed=None)
                #pos = nx.circular_layout(G=subgraph, scale=1, center=None, dim=2)
                #pos = nx.shell_layout(G=subgraph, nlist=None, scale=1, center=None, dim=2, rotate=None)
                #pos = nx.spiral_layout(G=subgraph, scale=1, center=None, dim=2, resolution=1, equidistant=False)
                #pos = nx.kamada_kawai_layout(G=subgraph, scale=1, center=None, dim=2, pos=None, weight='freq')

                # Set node size and color
                node_size = [value['node_size'] for key, value in subgraph.nodes.items()]
                node_color = [int(value['modularity']) for key, value in subgraph.nodes.items()]

                # Set edge width
                edge_width = [value['edge_width'] for key, value in subgraph.edges.items()]

                # Draw graph with color change
                nx.draw_networkx(subgraph, pos=pos, node_size=node_size, node_color=node_color, edge_color='gray', width=edge_width, with_labels=False, cmap=plt.cm.rainbow , vmin=0, vmax=len(self.node_modularity)-1)
                nx.draw_networkx_nodes(subgraph, pos=pos, node_size=node_size, edgecolors='black', node_color=node_color, cmap=plt.cm.rainbow , vmin=0, vmax=len(self.node_modularity)-1)
                
                # Add scaled font
                for node, (x, y) in pos.items():
                    text(x, y, node, fontsize=subgraph.nodes[node]['node_size']/100, ha='center', va='center', weight='bold', color='black')

                # Save graph as png
                plt.savefig(f"{self.save_path}/{folder_name}/community.png")
                plt.clf()

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
                #df = pd.DataFrame(columns=["conformity","title","DOI"])
                #total_pagerank = sum([value['pagerank'] for key, value in subgraph.nodes.data()])
                #for metadata in self.metadata_list:
                #    title = metadata['title']
                #    doi = metadata['DOI']
                #    conformity = 0
                #    for node in subgraph.nodes:
                #        if node in title:
                #            conformity += subgraph.nodes[node]['pagerank']
                #    df = df.append({'DOI':doi, 'title':title, 'conformity':float(f"{conformity/total_pagerank*100:.2f}")}, ignore_index=True)
                #df = df.sort_values(by=['conformity'], ascending=False)
                #df.to_csv(f"{self.save_path}/{folder_name}/pagerank_doc.csv", index=False)

                # update progress bar
                pbar.update(1)