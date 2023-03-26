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
        self.research_structurization(freq="total")
        self.keyword_filter(percent=self.filter_percent)
        self.community_labeling()
        self.save_graph()

    def graph_construct(self,freq):
        self.G = nx.Graph()
        node_list = []
        for key, value in self.node_feature.items():
            node_list.append((key,{"freq":value["year"][freq],"NER":value["NER"],"pagerank":None, "community":None}))
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

    def research_structurization(self, freq):
        def pagerank(self, freq):
            node_pagerank = nx.pagerank(self.G, alpha=0.85, max_iter=20, tol=1e-06, weight=freq, dangling=None)
            for key, value in self.G.nodes.data():
                self.G.nodes[key]['pagerank'] = node_pagerank[key]
                self.G.nodes[key]['size'] = node_pagerank[key] * 10000
            print("Pagerank calculation finished")

        def modularity(self, G, freq):
            # calculate modularity using Louvain method
            node_modularity = nx.algorithms.community.louvain_communities(G, weight=freq, resolution=self.community_resolution, seed=self.community_seed)
            #node_modularity = nx.algorithms.community.greedy_modularity_communities(G, weight=freq)
            # sorting community by size
            node_modularity = sorted(node_modularity, key=len, reverse=True)

            return node_modularity

        # calculate pagerank
        pagerank(self, freq=freq)

        # 1. calculate pagerank sum of graph
        total_pagerank = sum([self.G.nodes[node]['pagerank'] for node in self.G.nodes])

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
                    community_pagerank = sum([G.nodes[node]['pagerank'] for node in community])
                    print(f"community {idx} pagerank: {community_pagerank} ({community_pagerank/total_pagerank*100}%)")

                    # if pagerank sum of community is less than 10% of total pagerank, stop modularization and append graph to modularized_graphs
                    if community_pagerank/total_pagerank < 0.05:
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

    def community_labeling(self):
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

    def save_graph(self):
        # Save graph into gexf
        nx.write_gexf(self.G, f"{self.save_path}/graph.gexf")

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

                # Save graph into gexf
                nx.write_gexf(self.G, f"{self.save_path}/{folder_name}/community.gexf")

                # Save graph data as json
                with open(f"{self.save_path}/{folder_name}/community.json", "w") as f:
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