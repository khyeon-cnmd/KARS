# Todo
## 1. Metadata collection
    * 검색 엔진의 다양화 (Web of science, science direct 등)

## 2. Graph data extraction
    * 분리된 Keyword 에 물질, 공정, 구조, 물성, 성능의 Labeling 을 할 수 있다면? -> 어떤 물성과 연결성이 높은 물질 군의 확인 등 Graph 기반 분석이 가능할 것  <- 어떻게? WIKIPedia 학습 모델을 쓸 수 있으려나..
    -> 한번 라벨링을 해보고 유의미한 내용 또는 연결관계를 얻을 수 있는 지 테스트 해보는건?
    * keyword 추출 시 Vectorizer 종류를 Count 와 TF-IDF 선택 가능 -> 보류.. Graph network 에서 어떻게 작용할 지 모르겠음
    * node feature 에 추가할 수 있는 데이터: 논문 인용 수, ...
    * 단어간 연결관계를 통한 방향성 네트워크 구축 가능성

## 3. Graph network construction
    * 키워드 간 유사도 계산 -> 중복 키워드를 줄이자. (ReRAM RRAM -> 합치자) <======= similarity 이론을 써도, Clustering 처럼 결과가 나타나기 때문에 불가능한 듯 함.
    * Community 의 subgraph를 계속 그려도 노드 사이즈가 더이상 작아지지 않는 경우 -> 무한히 loop 를 돌게 됨. limit 필요

## 4. Research trend analysis
    * 연도별 그래프 추출 -> Interactive 한 그래프 개형 변화 확인
    * Gaussian fitting 을 개선하는 new fitting curve
    * 각 커뮤너티를 자연어 처리를 활용해 자동으로 정의하는 모델 -> description 과 같은 요약 모델?
    
## 5. Keyword to docs
    * 그래프간 유사도를 활용해서, subgraph 들과 논문 제목 graph 와의 유사도 비교 
    * 논문 제목으로 만든 subgraph 가 전체 graph 중 어떤 keyword cluster graph 에 매칭하는지 점수로 판단
    
## 6. NLP-based research overviewing
    * 재료 공학적 지식에 기반해 수집된 데이터를 분석하는 모듈
    * 커뮤너티 별 주요 키워드를 포함하는 논문을 Metadata 의 제목으로부터 선별하는 모델 -> 이후 해당 키워드에 대해서 그래프 재추출?
    * 선별된 논문을 자동으로 다운로드해 텍스트 및 이미지 데이터를 저장하는 모듈

# Done
## 1. Metadata collection
    * CrossrefAPI 기반의 Metadata 수집 기능

## 2. Graph data extraction
    * Title, abstract 텍스트의 전처리 필요성 (<sub> 같은 HTML 코드가 연결성 방해함)
    * Preprocessing 함수를 추가 -> Year, text filtering 에 따른 metadata 개수 변화 + text preprocessing 포함
    * 논문 title 및 abstract 중 그래프 화 대상 선택 가능
    * ngram 크기 조절을 통해 keyword 길이 조절 가능

## 3. Graph network construction
    * Pagerank 기반의 중요 키워드 추출
    * Louvain modularity 기반의 keyword clustering
    * 키워드의 Pagerank Limit을 통해 그래프 내 주요 키워드 선별 기능
    * 전체 Keyword frequencies 대비 커뮤니티의 Keyword frequencies를 기준으로 그래프 내 비율 나타냄
    * PYvis 기반의 Interactive graph visualization 구현 (ref: https://towardsdatascience.com/making-network-graphs-interactive-with-python-and-pyvis-b754c22c270, https://visjs.github.io/vis-network/docs/network/)

## 4. Research trend analysis
    * subgraphs 에 대한 trend analysis 자동화
    * Community_year_trend plot 에서 Outlier 필터링 기능 추가
    * Gaussian fitting 에서 얻은 PLC 라벨링 추가
    * subgraph 의 community_trend_year 가 전체 대비가 아니라, subgraph 대비로 되어 있음... 고쳐야함.

## 5. Keyword to docs
    * 커뮤니티 별 키워드의 Pagerank score sum 을 기준으로, 논문 제목의 점수를 매겨 분류

    
## 6. NLP-based research overviewing
    * 각 클러스터 별 논문 추출 후, 논문 본문들을 Semantic network 형성 -> clustering -> Dictionary based NER 라벨링
    * 