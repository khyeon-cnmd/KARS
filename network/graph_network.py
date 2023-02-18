import os 
import shutil
import networkx as nx
import json
import pandas as pd
from matplotlib.pyplot import figure, text
import matplotlib.pyplot as plt
from tqdm import tqdm

class graph_network:
    def __init__(self, save_path, DB_name):
        self.save_path = f"{save_path}/{DB_name}"
        # delete folder containing % sign
        for path, dirs, file in os.walk(self.save_path):
            for dir in dirs:
                if "%" in dir:
                    shutil.rmtree(f"{path}/{dir}")

        with open(f"{self.save_path}/node_feature.json", "r", encoding="utf-8") as f:
            self.node_feature = json.load(f)
        with open(f"{self.save_path}/edge_feature.json", "r", encoding="utf-8") as f:
            self.edge_feature = json.load(f)   
        #for total
        self.graph_construct(freq="total")    
        self.pagerank(freq="total")
        self.modularity(freq="total")
        self.node_size()
        self.edge_width()
        
    def __call__(self, save_path):
        self.save_path = save_path
        with open(f"{save_path}/subgraph.json", "r", encoding="utf-8") as f:
            subgraph = json.load(f)
        self.G = nx.node_link_graph(subgraph)
        self.modularity(freq="total")
        self.edge_width()
        self.save_graph()

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
        
        # Save graph data as json
        with open(f"{self.save_path}/graph.json", "w") as f:
            json.dump(nx.node_link_data(self.G), f, indent=4)

    def pagerank(self, freq):
        node_pagerank = nx.pagerank(self.G, alpha=0.85, max_iter=20, tol=1e-06, weight=freq, dangling=None)
        for key, value in self.G.nodes.data():
            self.G.nodes[key]['pagerank'] = node_pagerank[key]

    def modularity(self, freq):
        self.node_modularity = nx.algorithms.community.louvain_communities(self.G, weight=freq, resolution=1.0)
        for key, value in self.G.nodes.data():
            for community in self.node_modularity:
                if key in community:
                    if self.G.nodes[key]['modularity'] == None:
                        self.G.nodes[key]['modularity'] = f"{self.node_modularity.index(community)}"
                    else:
                        self.G.nodes[key]['modularity'] = f"{self.G.nodes[key]['modularity']}_{self.node_modularity.index(community)}"
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

    def save_graph(self):
        # Set overall figure size
        plt.figure(figsize=(20,20), dpi=300).tight_layout()

        with tqdm(total=len(self.node_modularity), desc="Save graph") as pbar:
            for community in self.node_modularity:
                # make folder for each community
                community_index = self.node_modularity.index(community)
                percent_of_community = round(len(community)/self.total_nodes*100, 2)
                folder_name = f"{community_index} ({percent_of_community}%)" 
                print(f"Community {community_index} has {len(community)} nodes, {percent_of_community}% of total nodes")
                if not os.path.exists(f"{self.save_path}/{folder_name}"):
                    os.makedirs(f"{self.save_path}/{folder_name}")
                
                # make subgraph by community
                def filter_node(n1):
                    return n1 in community
                subgraph = nx.subgraph_view(self.G, filter_node=filter_node)

                # Graph info
                #print("Nodes: ", subgraph.nodes.data())
                #print("Edges: ", subgraph.edges.data())

                # Set layout and position of nodes. k is distance btw nodes
                #pos = nx.spring_layout(G=subgraph, k=5, weight='freq', iterations=100, scale=1, center=None, dim=2, seed=None)
                pos = nx.circular_layout(G=subgraph, scale=1, center=None, dim=2)
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
                with open(f"{self.save_path}/{folder_name}/subgraph.json", "w") as f:
                    json.dump(nx.node_link_data(subgraph), f, indent=4)

                # Save node ranks as csv
                df = pd.DataFrame()
                df['node'] = [key for key, value in subgraph.nodes.data()]
                df['pagerank'] = [value['pagerank'] for key, value in subgraph.nodes.data()]
                df = df.sort_values(by=['pagerank'], ascending=False)
                df.to_csv(f"{self.save_path}/{folder_name}/pagerank.csv", index=False)

                # update progress bar
                pbar.update(1)