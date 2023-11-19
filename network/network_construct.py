import os
import json
import networkx as nx
from tqdm import tqdm


class network_construction:
    def __init__(self, DB_path):
        self.DB_path = f"{DB_path}/database"
        self.save_path = f"{DB_path}/KARS"

    def network_construct(self):
        print("network_construct by co-occurrence")
        # 1. database 내에 있는 모든 paper_index 에 대해서
        for paper_index in tqdm(os.listdir(f"{self.DB_path}")):
            # 1-1. KARS.json 파일이 있는지 확인
            if not os.path.isfile(f"{self.DB_path}/{paper_index}/KARS.json"):
                continue

            # 1-2. KARS.json 읽기
            KARS_dict = json.load(open(f"{self.DB_path}/{paper_index}/KARS.json", "r", encoding="utf-8-sig"))

            # 1-3. 비방향성 그래프 생성
            G = nx.Graph()

            # 1-4. keyword_list 에 대해 co-occurrence edge 구축
            keyword_list = KARS_dict["keyword_tokenization"]["title"]
            for i, keyword in enumerate(keyword_list):
                for j, keyword2 in enumerate(keyword_list):
                    if i <= j:
                        continue
                    if G.has_edge(keyword, keyword2):
                        G[keyword][keyword2]["weight"] += 1
                    else:
                        G.add_edge(keyword, keyword2, weight=1)

            # 1-5. co-occurrence node 구축
            year = str(KARS_dict["cover_data"]["published_date"][0])
            for keyword in keyword_list:
                if not keyword in G.nodes:
                    G.add_node(keyword)

                if "weight" in G.nodes[keyword]:
                    G.nodes[keyword]["weight"] += 1
                else:
                    G.nodes[keyword]["weight"] = 1
                
                if year in G.nodes[keyword]:
                    G.nodes[keyword][year] += 1
                else:
                    G.nodes[keyword][year] = 1

            # 1-6. save network
            nx.write_gexf(G, f"{self.DB_path}/{paper_index}/KARS.gexf")

    def network_integrate(self):
        print(f"network integrate")
        # 1. 비방향성 그래프 생성
        Total_G = nx.Graph()

        # 2. database 내에 있는 모든 paper_index 에 대해서
        for paper_index in tqdm(os.listdir(f"{self.DB_path}")):
            if os.path.isfile(f"{self.DB_path}/{paper_index}/KARS.gexf"):
                G = nx.read_gexf(f"{self.DB_path}/{paper_index}/KARS.gexf")

                # 노드 속성 합산
                for node_id, node_data in G.nodes(data=True):
                    if Total_G.has_node(node_id):
                        # 기존 노드가 있는 경우, 속성 값을 합산
                        existing_node = Total_G.nodes[node_id]
                        for attr_name, attr_value in node_data.items():
                            if not attr_name == "label":
                                existing_node[attr_name] = existing_node.get(attr_name, 0) + attr_value
                    else:
                        # 기존 노드가 없는 경우, 노드와 속성 추가
                        Total_G.add_node(node_id, **node_data)

                # 엣지 속성 합산
                for source, target, edge_data in G.edges(data=True):
                    if Total_G.has_edge(source, target):
                        # 기존 엣지가 있는 경우, 속성 값을 합산
                        existing_edge = Total_G[source][target]
                        for attr_name, attr_value in edge_data.items():
                            existing_edge[attr_name] = existing_edge.get(attr_name, 0) + attr_value
                    else:
                        # 기존 엣지가 없는 경우, 엣지와 속성 추가
                        Total_G.add_edge(source, target, **edge_data)

        # Reset Id of edges
        for i, edge in enumerate(Total_G.edges()):
            Total_G[edge[0]][edge[1]]['id'] = i

        # Add weight_rev for edges
        for edge in Total_G.edges():
            Total_G[edge[0]][edge[1]]['weight_rev'] = 1 / Total_G[edge[0]][edge[1]]['weight']

        nx.write_gexf(Total_G, f"{self.save_path}/KARS.gexf")