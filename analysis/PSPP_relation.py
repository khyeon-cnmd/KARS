import networkx as nx
import pandas as pd
from tqdm import tqdm

class PSPP_relation:
    def __init__(self, DB_path):
        self.DB_path = DB_path + "/KARS"
        self.tree_G = nx.read_gexf(self.DB_path + "/PSPP_abstract_network.gexf")
        # save weight_rev 
        for edge in self.tree_G.edges():
            self.tree_G.edges[edge]["weight_rev"] = 1 / self.tree_G.edges[edge]["weight"]
            
        print(self.tree_G)
        pass

    def community_detection(self):
        # using louvain's modularity maximization
        node_modularity = nx.algorithms.community.louvain_communities(self.tree_G, weight="weight", resolution=1, seed=42)
        node_modularity = sorted(node_modularity, key=len, reverse=True)
        modularity_dict = {}
        for i, community in enumerate(node_modularity):
            modularity_dict[i] = community

        # community classification
        for i, community in modularity_dict.copy().items():
            # community의 node 를 weight 로 sorting
            community = [node for node, weight in sorted(self.tree_G.degree(community), key=lambda x: x[1], reverse=True)]
            name = community[0]
            print("Community", i, ":", len(community), "nodes")

            # print 상위 10개 노드
            # for node in community[:20]:
            #     # get degree
            #     print(self.tree_G.nodes[node]["label"], self.tree_G.degree[node])
            
            # get user input
            #name = input("What is the community? ")

            # substitute the key name of i into name
            modularity_dict[name] = modularity_dict[i]
            del modularity_dict[i]

            print()

        #  depth separation
        depth_dict = {}
        for node in self.tree_G.nodes:
            depth = self.tree_G.nodes[node]["depth"]
            if depth not in depth_dict:
                depth_dict[depth] = []
            depth_dict[depth].append(node)

        # in depth, community classification
        self.depth_modularity_dict = {}
        for depth in depth_dict:
            self.depth_modularity_dict[depth] = {}
            for i, community in modularity_dict.items():
                if not i in self.depth_modularity_dict[depth]:
                    self.depth_modularity_dict[depth][i] = []
                for node in depth_dict[depth]:
                    if node in community:
                        self.depth_modularity_dict[depth][i].append(node)
                if not self.depth_modularity_dict[depth][i]:
                    del self.depth_modularity_dict[depth][i]

        pass

    def search_keyword(self):
        # print tree structure
        for depth in self.depth_modularity_dict.keys():
            df = pd.DataFrame()
            for community in self.depth_modularity_dict[depth].keys():
                i = 0
                for node in self.depth_modularity_dict[depth][community]:
                    df.loc[i,community] = node
                    i += 1
            print(df)
            print(f"depth: {depth} --------------------------------------------------------------------------------")
        while 1:
            # get user input
            keyword = input("What keyword do you want to search? ")

            # find all edges of keyword
            edge_list = []
            for edge in self.tree_G.edges:
                if keyword in edge:
                    edge_list.append(edge)
            
            # print keyword->node and node->keyword
            df = pd.DataFrame(columns=["child", "child weight", "parent", "parent weight"])
            parent_index = 0
            parent_weight_index = 0
            child_index = 0
            child_weight_index = 0
            for edge in edge_list:
                if edge[1] == keyword:
                    df.loc[child_index, "child"] = edge[0]
                    df.loc[child_weight_index, "child weight"] = self.tree_G.edges[edge]["weight"]
                    child_index += 1
                    child_weight_index += 1
                else: 
                    df.loc[parent_index, "parent"] = edge[1]
                    df.loc[parent_weight_index, "parent weight"] = self.tree_G.edges[edge]["weight"]
                    parent_index += 1
                    parent_weight_index += 1

            # child 를 child weight 로 sorting, parent 를 parent weight 로 sorting
            df_child = df[["child", "child weight"]].sort_values(by="child weight", ascending=False)
            df_parent = df[["parent", "parent weight"]].sort_values(by="parent weight", ascending=False)
            df = pd.concat([df_child, df_parent], axis=1)
            # index reset
            df.reset_index(drop=True, inplace=True)
            
            print(df)
            print()

    def search_path(self, edge_weight_limit, target_node):
        # 1. limit edges by weight
        for edge in self.tree_G.copy().edges:
            if self.tree_G.edges[edge]["weight"] < edge_weight_limit:
                self.tree_G.remove_edge(*edge)

        # 2. remove nodes that having no edges
        for node in self.tree_G.copy().nodes:
            if not self.tree_G.degree[node]:
                self.tree_G.remove_node(node)

        # 3. find high weighted paths
        self.path_list = []
        for node in tqdm(self.tree_G.nodes):
            try:
                path = nx.shortest_path(self.tree_G, source=node, target=target_node, weight="weight_rev")
                self.path_list.append(path)
            except:
                pass

        # 4. sort path_list by path length
        #self.path_list = sorted(self.path_list, key=lambda x: sum([self.tree_G.edges[edge]["weight"] for edge in zip(x[:-1], x[1:])]), reverse=True)
        self.path_list = sorted(self.path_list, key=lambda x: len(x), reverse=False)

        # 5. print path_list
        for path in self.path_list:
            print(path, sum([self.tree_G.edges[edge]["weight"] for edge in zip(path[:-1], path[1:])]))

        

