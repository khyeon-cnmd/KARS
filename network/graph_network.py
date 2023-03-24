import os 
import shutil
import numpy as np
import networkx as nx
import netgraph
import pyvis
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
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
        self.modularity(freq="total")
        self.save_graph()
        self.save_subgraph()
        
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
            node_list.append((key,{"freq":value["year"][freq],"NER":value["NER"],"pagerank":None,"modularity":None}))
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

    def synonyms(self):
        # finding synonyms using edge linkage information
        # if two nodes are connected by more than 50% of total edges, they are synonyms
        def pagerank(G, freq):
            node_pagerank = nx.pagerank(G, alpha=0.85, max_iter=20, tol=1e-06, weight=freq, dangling=None)
            for key, value in G.nodes.data():
                G.nodes[key]['pagerank'] = node_pagerank[key]

        def weighted_simrank(G, C=0.8, max_iter=100, eps=1e-4):
            # Initialize SimRank scores
            nodes = list(G.nodes())
            simrank = np.zeros((len(nodes), len(nodes)))
            for i, u in enumerate(nodes):
                simrank[i, i] = 1.0

            # Compute SimRank scores
            for iter in tqdm(range(max_iter)):
                prev_simrank = simrank.copy()
                for i, u in enumerate(nodes):
                    for j, v in enumerate(nodes):
                        if i == j:
                            continue
                        neighbors_u = list(G.neighbors(u))
                        neighbors_v = list(G.neighbors(v))
                        sum = np.sum(prev_simrank[[nodes.index(n1) for n1 in neighbors_u], :][:, [nodes.index(n2) for n2 in neighbors_v]])
                        simrank[i, j] = (C * sum) / (len(neighbors_u) * len(neighbors_v))
                        simrank[i, j] += (1 - C) * G.nodes[u]["pagerank"] * G.nodes[v]["pagerank"]
                # Check convergence
                err = np.sum(np.abs(simrank - prev_simrank))
                if err < eps:
                    break

            return simrank

        def keyword_filter(G, percent):
            # sort self.G.nodes by pagerank
            node_list = sorted(G.nodes, key=lambda x: G.nodes[x]['pagerank'], reverse=True)
            x = [i for i in range(0, len(node_list)+1)]
            y = [sum([G.nodes[node]['pagerank'] for node in node_list[:i]]) for i in range(1, len(node_list)+1)]
            y.insert(0,0)

            # filter in nodes until integrated pagerank is 80%
            integrated_pagerank = 0
            total_pagerank = sum([G.nodes[node]['pagerank'] for node in G.nodes])
            alive_nodes = []
            for node in node_list:
                integrated_pagerank += G.nodes[node]['pagerank']
                alive_nodes.append(node)
                if integrated_pagerank/total_pagerank*100 > percent:
                    break
            remove_nodes = [node for node in G.nodes if node not in alive_nodes]

            # filter out nodes
            for node in remove_nodes:
                G.remove_node(node)

            # print graph information
            print("Keyword Filtered Graph information")
            print("Node number: ", G.number_of_nodes())
            print("Edge number: ", G.number_of_edges())

            return G

        # filter out Top 20% nodes
        G_20 = self.G.copy()
        G_20 = keyword_filter(G_20, percent=20)

        # calculate pagerank
        pagerank(G_20, freq="freq")
        print(G_20.nodes.data())

        # calculate SimRank similarity
        simrank = weighted_simrank(G_20)
        node_list = sorted(G_20.nodes, key=lambda x: G_20.nodes[x]['pagerank'], reverse=True)
        for node in node_list:
            print(node, sorted(zip(node_list, simrank[node_list.index(node)]), key=lambda x: x[1], reverse=True)[1:10])


        # find synonyms using SimRank similarity
        # SimRank similarity is a measure of similarity between two nodes in a graph
        #for node in node_list:
        #    simrank = nx.simrank_similarity(self.G, source=node, target=None, importance_factor=0.1, max_iterations=1000, tolerance=0.0001)
        #    print(node, sorted(simrank.items(), key=lambda x: x[1], reverse=True)[:10])            
        #    #panther = nx.panther_similarity(self.G, source=node, k=5, path_length=5, c=0.5, delta=0.1, eps=None)
        #    #print(node, sorted(panther.items(), key=lambda x: x[1], reverse=True)[:10])

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

    def modularity(self, freq):
        self.node_modularity = nx.algorithms.community.louvain_communities(self.G, weight=freq, resolution=self.community_resolution, seed=self.community_seed)
        # sorting community by size
        self.node_modularity = sorted(self.node_modularity, key=len, reverse=True)
        for key, value in self.G.nodes.data():
            for idx, community in enumerate(self.node_modularity):
                if key in community:
                    self.G.nodes[key]['modularity'] = idx
                    break
        
        print("Modularity information")
        print("Community number: ", len(self.node_modularity))
        print("Community size: ", [len(community) for community in self.node_modularity])
        print("Community pagerank: ", [sum([self.G.nodes[node]['pagerank'] for node in community]) for community in self.node_modularity])

    def save_graph(self):
        # Set colors
        colors = plt.cm.rainbow
        node_color = {node: colors(self.G.nodes[node]['modularity']/len(self.node_modularity)) for node in self.G.nodes}

        # make graph plot by netgraph
        #netgraph.Graph(self.G, 
        #node_color=node_color,
        #node_edge_width=0,
        #edge_alpha=0.1,
        #node_layout="community", node_layout_kwargs=dict(node_to_community=self.G.nodes.data('modularity')),
        #edge_layout="bundled"
        #)
        #plt.savefig(f"{self.save_path}/graph.png")


        # make interactive html based graph network using pyvis
        nt = pyvis.network.Network(height="1000px", width="100%", bgcolor="#ffffff", font_color="#000000", layout={}, filter_menu=True, select_menu=True) 
        nt.from_nx(self.G)
        # change font size of node by pagerank
        for node in nt.nodes:
            for i, community in enumerate(self.node_modularity):
                if node["id"] in community:
                    color = colors(i/len(self.node_modularity))
                    color = matplotlib.colors.to_hex(color)
                    node["value"] = self.G.nodes[node["id"]]["pagerank"]
                    node["title"] = f"Node: {node['id']}<br>Pagerank: {self.G.nodes[node['id']]['pagerank']}<br>Modularity: {i}"
                    node["color"] = {"background": color, "highlight": "#F2DC23", "hover": color, "border": "#000000"}
                    break
        # change width of edge by freq
        for edge in nt.edges:
            edge["value"] = edge["freq"]
            edge["color"] = {"color": "#808080", "highlight": "#F2DC23", "hover": "#F2DC23"} 
            edge["arrows"] = {"to": {"enabled": True, "scaleFactor": 1}}
            edge["selectionWidth"] = 3
        # Make a group
        
#
        nt.force_atlas_2based(gravity=-50, central_gravity=0.3, spring_length=100, spring_strength=0.08, damping=0.4, overlap=1)
        nt.show_buttons(filter_=["physics","interaction","nodes","edges","layout"])
        nt.options.layout = {"improvedLayout":True, "clusterThreshold":500}
        nt.options.interaction.navigationButtons = True
        nt.options.interaction.keyboard = True
        nt.options.physics.enabled = False
        nt.write_html(f"{self.save_path}/graph.html", local=True, notebook=False, open_browser=True)
        
        # Save graph data as json
        with open(f"{self.save_path}/graph.json", "w") as f:
            json.dump(nx.node_link_data(self.G), f, indent=4)

        # Save node ranks as csv1
        df = pd.DataFrame()
        df['node'] = [key for key, value in self.G.nodes.data()]
        df['pagerank'] = [value['pagerank'] for key, value in self.G.nodes.data()]
        df = df.sort_values(by=['pagerank'], ascending=False)
        df.to_csv(f"{self.save_path}/pagerank.csv", index=False)

        print("Saving graph data is done")

    def save_subgraph(self):
        # Set colors
        colors = plt.cm.rainbow
        with tqdm(total=len(self.node_modularity), desc="Save graph") as pbar:
            for i, community in enumerate(self.node_modularity):
                # make subgraph by community
                def filter_node(n1):
                    return n1 in community
                subgraph = nx.subgraph_view(self.G, filter_node=filter_node)

                # make folder for each community
                community_index = self.node_modularity.index(community)
                subgraph_pagerank = sum([subgraph.nodes[node]['pagerank'] for node in subgraph.nodes])
                percent_of_community = round(subgraph_pagerank/self.total_pagerank*100, 2)
                folder_name = f"{community_index} ({percent_of_community}%)" 
                print(f"Community {community_index} has {len(community)} nodes, {percent_of_community}% of total nodes")
                if not os.path.exists(f"{self.save_path}/{folder_name}"):
                    os.makedirs(f"{self.save_path}/{folder_name}")

                # make interactive html based graph network using pyvis
                nt = pyvis.network.Network(height="1000px", width="100%", bgcolor="#ffffff", font_color="#000000", layout={}) #directed=False, select_menu=True, filter_menu=True, , width="80%", 
                nt.from_nx(subgraph)
                # change font size of node by pagerank
                for node in nt.nodes:
                    node["value"] = node["pagerank"]
                    color = colors(i/len(self.node_modularity))
                    color = matplotlib.colors.to_hex(color)
                    node["color"] = {"background": color, "highlight": "#F2DC23", "hover": color, "border": "#000000"}
                # change width of edge by freq
                for edge in nt.edges:
                    edge["value"] = edge["freq"]
                    edge["color"] = {"color": "#808080", "highlight": "#F2DC23", "hover": "#F2DC23"} 
                    edge["arrows"] = {"to": {"enabled": True, "scaleFactor": 1}}
                    edge["selectionWidth"] = 3
                #nt.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=250, spring_strength=0.001, damping=0.09, overlap=1)
                nt.force_atlas_2based(gravity=-50, central_gravity=0.3, spring_length=100, spring_strength=0.08, damping=0.4, overlap=1)
                nt.show_buttons(filter_=["physics","interaction","nodes","edges","layout"])
                nt.options.layout = {"improvedLayout":True, "clusterThreshold":500}
                #nt.options.autoResize = True
                nt.options.interaction.navigationButtons = True
                nt.options.interaction.keyboard = True
                nt.options.physics.enabled = False
                nt.write_html(f"{self.save_path}/{folder_name}/community.html", local=True, notebook=False, open_browser=True)

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
                    title = metadata['title']
                    doi = metadata['DOI']
                    conformity = 0
                    for node in subgraph.nodes:
                        if node in title:
                            conformity += subgraph.nodes[node]['pagerank']
                    df = df.append({'DOI':doi, 'title':title, 'conformity':float(f"{conformity/total_pagerank*100:.2f}")}, ignore_index=True)
                df = df.sort_values(by=['conformity'], ascending=False)
                df.to_csv(f"{self.save_path}/{folder_name}/pagerank_doc.csv", index=False)

                # update progress bar
                pbar.update(1)

        print("Saving subgraph data is done")