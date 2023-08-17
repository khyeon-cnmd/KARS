import os
import networkx as nx
import pandas as pd
from tqdm import tqdm

class PSPP_relation:
    def __init__(self, DB_path, para_type):
        self.KARS_path = DB_path + "/KARS"
        self.DB_path = DB_path + "/database"
        self.tree_G = nx.read_gexf(self.KARS_path + f"/PSPP_{para_type}_tree.gexf")
        # save weight_rev 
        for edge in self.tree_G.edges():
            self.tree_G.edges[edge]["weight_rev"] = 1 / self.tree_G.edges[edge]["weight"]
            
        print(self.tree_G)
        pass

    def construct_PSPP_tree(self, para_type, node_weight_limit, edge_weight_limit):
            print("Construct PSPP tree")

            # Load PSPP_abstract_network
            G = nx.read_gexf(f"{self.DB_path}/KARS/PSPP_{para_type}_network.gexf")
            G_copy = G.copy()
            print(G_copy)
            tree_G = nx.DiGraph()

            # 3. filter nodes
            for node in G_copy.copy().nodes():
                # 길이가 1글자면 제외
                if len(node) <= 1:
                    G_copy.remove_node(node)
                    continue

                # 빈도수가 3이하면 제외
                if G_copy.nodes[node]["weight"] <= node_weight_limit:
                    G_copy.remove_node(node)
                    continue

            # 4. filter edges
            for edge in G_copy.copy().edges():
                # 빈도수가 3이하면 제외
                if G_copy.edges[edge]["weight"] <= edge_weight_limit:
                    G_copy.remove_edge(edge[0], edge[1])
                    if G_copy.degree(edge[0]) == 0:
                        G_copy.remove_node(edge[0])
                    if G_copy.degree(edge[1]) == 0:
                        G_copy.remove_node(edge[1])
                    continue
            print(G_copy)
            
            # find root nodes 
            print("Find root nodes")
            root_node_list = [] 
            for node in tqdm(G_copy.nodes()):
                root_node = False
                for node2 in G_copy.nodes():
                    if node == node2:
                        continue

                    if not G.has_edge(node, node2):
                        node_to_node2_weight = 0
                    else:
                        node_to_node2_weight = G.edges[node, node2]["weight"]

                    if not G.has_edge(node2, node):
                        node2_to_node_weight = 0
                    else:
                        node2_to_node_weight = G.edges[node2, node]["weight"]

                    if node_to_node2_weight == 0 and node2_to_node_weight == 0:
                        continue

                    # root_node 가 모든 root_node2 대비 좌->우 인접 빈도가 더 크고 root_node 의 빈도가 root_node2 의 빈도보다 크면 진짜 root node
                    if node_to_node2_weight > node2_to_node_weight and G.nodes[node]["weight"] < G.nodes[node2]["weight"]:
                        root_node = False
                        break
                    else:
                        root_node = True

                if root_node == True:
                    root_node_list.append(node)
            
            # root node 중, node weight 로 최후의 root node 선정
            root_node_list = sorted(root_node_list, key=lambda x: G.nodes[x]["weight"], reverse=True)
            parent_node_list = root_node_list[:1]
            tree_G.add_node(root_node_list[0], **G.nodes[root_node_list[0]])
            tree_G.nodes[root_node_list[0]]["depth"] = 0

            # construct tree
            while parent_node_list:
                new_parent_node_list = []
                print(f"parent_node_list: {parent_node_list}")
                for parent_node in tqdm(parent_node_list):
                    for child_node in G_copy.nodes():
                        # 자기 자신은 제외
                        if child_node == parent_node:
                            continue

                        # shortest path 탐색
                        try:
                            path = nx.shortest_path(G_copy, source=child_node, target=parent_node, weight='weight_rev')
                        except:
                            continue

                        # path 길이 확인
                        if not len(path) == 2:
                            continue

                        # Tree G 에 추가
                        if not child_node in parent_node_list:
                            new_parent_node_list.append(child_node)
                            tree_G.add_node(child_node, **G.nodes[child_node])
                            tree_G.nodes[child_node]["depth"] = tree_G.nodes[parent_node]["depth"] - 1
                        tree_G.add_edge(child_node, parent_node, **G.edges[child_node, parent_node])

                # remove nodes of G_copy 
                for node in parent_node_list:
                    G_copy.remove_node(node)

                # make new_parent_node_list into parent_node_list
                parent_node_list = list(set(new_parent_node_list))

            # save tree
            nx.write_gexf(tree_G, f"{self.DB_path}/KARS/PSPP_{para_type}_tree.gexf")

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

    def search_path(self, edge_weight_limit, target_node, except_node_list):
        # 1. limit edges by weight
        for edge in self.tree_G.copy().edges:
            if self.tree_G.edges[edge]["weight"] < edge_weight_limit:
                self.tree_G.remove_edge(*edge)

        # 2. remove nodes that having no edges
        for node in self.tree_G.copy().nodes:
            if not self.tree_G.degree[node]:
                self.tree_G.remove_node(node)

        # 3. remove except_node_list
        for node in except_node_list:
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
        self.path_list = sorted(self.path_list, key=lambda x: sum([self.tree_G.edges[edge]["weight"] for edge in zip(x[:-1], x[1:])]), reverse=False)
        #self.path_list = sorted(self.path_list, key=lambda x: len(x), reverse=False)

        # 5. print path_list
        for path in self.path_list:
            print(path, sum([self.tree_G.edges[edge]["weight"] for edge in zip(path[:-1], path[1:])]))

    def search_path_by_paper(self, source_node, target_node):
        # 1. find for papers
        source_target_path = {}
        Total_G = nx.DiGraph()
        for index in tqdm(os.listdir(self.DB_path)):
            if os.path.isfile(f"{self.DB_path}/{index}/PSPP_abstract.gexf"):
                G = nx.read_gexf(f"{self.DB_path}/{index}/PSPP_abstract.gexf")

                # find shortest path from source_node to target_node
                try:
                    path = nx.shortest_path(G, source=source_node, target=target_node, weight="weight")
                    source_target_path[index] = path
                except:
                    continue

                # make a tree graph using Path
                for i in range(len(path)-1):
                    if not Total_G.has_edge(path[i], path[i+1]):
                        Total_G.add_edge(path[i], path[i+1], weight=G.edges[path[i], path[i+1]]["weight"], paper_index=index)
                    else:
                        Total_G.edges[path[i], path[i+1]]["weight"] += G.edges[path[i], path[i+1]]["weight"]
                        Total_G.edges[path[i], path[i+1]]["paper_index"] += f",{index}"

                for node in path:
                    depth = 1 / len(path) * path.index(node)
                    # if not Total_G.has_node(node):
                    #     Total_G.add_node(node, weight=G.nodes[node]["weight"], depth=i)
                    if "weight" not in Total_G.nodes[node]:
                        Total_G.nodes[node]["weight"] = G.nodes[node]["weight"]
                    else:
                        Total_G.nodes[node]["weight"] += G.nodes[node]["weight"]

                    if "depth" not in Total_G.nodes[node]:
                        Total_G.nodes[node]["depth"] = depth
                    elif i > Total_G.nodes[node]["depth"]:
                        Total_G.nodes[node]["depth"] = depth

        # Reset Id of edges
        for i, edge in enumerate(Total_G.edges()):
            Total_G[edge[0]][edge[1]]['id'] = i

        # 2. sort path_list by path length
        source_target_path = sorted(source_target_path.items(), key=lambda x: len(x[1]), reverse=False)

        # 3. print path_list
        for path in source_target_path:
            print(path)

        # save Total_G
        nx.write_gexf(Total_G, f"{self.KARS_path}/{source_node}_{target_node}.gexf")


