# Todo
## 1. Metadata collection
    * 검색 엔진의 다양화 (Web of science, science direct 등)

## 2. Graph data extraction
    * keyword 추출 시 Vectorizer 종류를 Count 와 TF-IDF 선택 가능 -> 보류.. Graph network 에서 어떻게 작용할 지 모르겠음
    * node feature 에 추가할 수 있는 데이터: 논문 인용 수, + Spacy 품사 분류 결과 -> Multi-keyword 분류에 활용할 수 도?
    * 단어간 연결관계를 통한 방향성 네트워크 구축 가능성
    * Abbrebiation 인식하여 대문자 유지 및 동의어 처리
    * 키워드 간 유사도 계산 -> 중복 키워드를 줄이자. (ReRAM RRAM -> 합치자) <======= similarity 이론을 써도, Clustering 처럼 결과가 나타나기 때문에 불가능한 듯 함.
    * Edge frequency 를 활용해서, single keywords 를 multi-keyword 화 할 수 있지 않을까? -> 이를 통해서 사용자는 Performance 등의 주요 키워드를 더 정확하게 확인 가능
    * title, Abstract 는 결국 PSPP 관계가 모두 담겨있는 종합적 데이터를 포함하다 보니, 각 Community 는 종합적 결과에 따른 주제로 나타나게 됨.
    따라서, PSPP 관계를 탐색하기 위해서는 NER 모델을 활용해 단어별 PSPP Labeling 을 하는 수 밖에 없음

## 3. Graph network construction
    * Modularity 를 통해 구해진 결과를 따로 저장 -> 이는 전체 데이터에 대한 종합적 결과로 research trend analysis 에 활용
    * Gephi toolkit 을 활용해 gexf 형식의 파일을 백그라운드로 변형해 Interactive graph로 출력하도록 하기!(https://github.com/jsundram/pygephi)

## 4. Research trend analysis
    * 연도별 그래프 추출 -> Interactive 한 그래프 개형 변화 확인
    * Gaussian fitting 을 개선하는 new fitting curve
    
## 5. GUI
    * 이미 데이터가 있는 경우, 코드 실행 단계에서 Output 만 읽도록 구성할 수 있을 듯?
    * Interactive graph 는 HTML 형식으로 출력한 것 보도록
    * 현존 데이터가 있더라도, Regeneration 할 수 있도록 구성
    * https://github.com/gephi/gephi-lite#readme

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
    * Chemical composition 을 인식하여 material 로 Labeling
    * Chemical composition 중 소문자로 존재하는 단어들을 동의어 처리 하기

## 3. Graph network construction
    * Pagerank 기반의 중요 키워드 추출
    * Louvain modularity 기반의 keyword clustering
    * 키워드의 Pagerank Limit을 통해 그래프 내 주요 키워드 선별 기능
    * 전체 Keyword frequencies 대비 커뮤니티의 Keyword frequencies를 기준으로 그래프 내 비율 나타냄
    * PYvis 기반의 Interactive graph visualization 구현 (ref: https://towardsdatascience.com/making-network-graphs-interactive-with-python-and-pyvis-b754c22c270, https://visjs.github.io/vis-network/docs/network/)
    * Gephi 형식으로 그래프 출력 기능 구현
    * Modularized cluster 를 Material, Processing, Structure, Property, Performance ,Other 로 Manual labeling 기능 추가
    * Recursively graph modularize 가 가능하게 구현

## 4. Research trend analysis
    * subgraphs 에 대한 trend analysis 자동화
    * Community_year_trend plot 에서 Outlier 필터링 기능 추가
    * Gaussian fitting 에서 얻은 PLC 라벨링 추가

## 5. GUI
    * Input field 를 통해 원하는 주제, 단계, 알고리즘으로 연구 동향을 분석하고 결과물을 출력하는 시스템 구축