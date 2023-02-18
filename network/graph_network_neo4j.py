from neo4j import GraphDatabase
import json

class Neo4j_operation:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

if __name__ == "__main__":
    Neo4j = Neo4j_operation("neo4j://141.223.167.14:7687/", "neo4j", "15926485Gg!")
    filename = "/home1/khyeon/Researches/2_Text_mining/KBRS/node_feature.json"


//Node feature.csv 불러오기
LOAD CSV WITH HEADERS FROM 'file:///node_feature.csv' AS row
CREATE (:keywords {name:row.keyword, total:toInteger(row.total)})

//Edge feature.csv 불러오기
LOAD CSV WITH HEADERS FROM 'file:///edge_feature.csv' AS row
WITH row, SPLIT(row.relation, '-') AS parts
MATCH (a:keywords {name:parts[0]}), (b:keywords {name:parts[1]}) 
CREATE (a)-[:co_occur {total:toInteger(row.total)}]->(b)

//PageRank 구하기
CALL gds.graph.project('KCN','keywords','co_occur',{relationshipProperties: 'total'})
CALL gds.pageRank.write('KCN', {maxIterations: 20, dampingFactor: 0.85, relationshipWeightProperty: 'total', writeProperty: 'pagerank'}) 

// Network modularity 구하기
CALL gds.louvain.write("KCN",{relationshipWeightProperty:'total', writeProperty:"community"})


//Pagerank 로 node filtering
MATCH (n:keywords)
WHERE n.pagerank > 0.03
RETURN n.name, n.pagerank

//community 목록 확인
MATCH (n:keywords)
RETURN DISTINCT n.community

// Community 에 따라서 node color 변경
// BY NATIONALITY
MATCH (n:keywords)
WITH DISTINCT n.community AS communities, collect(DISTINCT n) AS keywords
CALL apoc.create.addLabels(keywords, [apoc.text.upperCamelCase(communities)]) YIELD node
RETURN *