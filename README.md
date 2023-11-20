<p align="center"><img src="./examples/KARS_logo.png" width="250" height="250">

# Keyword-based Automatic Research Structurization (KARS)
과학 연구 분야의 서지 정보를 바탕으로 연구 키워드 추출, 연구 구조화, 연구 동향 분석을 자동화하여 연구자에게 연구의 공간적 형태 및 시간적 흐름을 시각적으로 제공하는 프로그램

# Installation
## 1. Github repository 로드
    git clone --recurse-submodules https://github.com/khyeon-cnmd/KARS.git

## 2. anaconda 설치
    conda create -n KARS python==3.10
    conda activate KARS

## 3. python libraries 설치
    pip install jsonlines
    pip install gradio==3.47.0
    pip install networkx[default]
    pip install tqdm
    pip install pandas
    pip install scipy
    pip install bokeh
    pip install spacy

## 3. Spacy 언어팩 설치
    python -m spacy download en_core_web_sm
    python -m spacy download en_core_web_trf

# Usage
KARS_GUI.py 실행 후, https://127.0.0.1:7860 에 접속해 관련 설정 후 제출

# Results
## 1. KARS.gexf
PageRank Algorithm을 통한 주요 키워드 선별, Louvain's Modularity를 통해 구축한 모듈화된 키워드 네트워크 결과로, 노드 크기를 통해 키워드 중요도, 노드 색을 통해 키워드 커뮤니티를 나타냄 (Gephi 프로그램을 통해 분석)

## 2. research_maturity.html
전체 커뮤니티의 연도에 따른 키워드 수의 변화 그래프. 제품 수명 주기 (PLC model)에 기반해 해당 연구 분야의 연구 성숙도를 평가
<p align="center"><img src="./examples/research_maturity.png" width="250" height="250">

## 3. community_year_trend.html
연구 커뮤니티 별 연도에 따른 키워드 분포의 변화 그래프. 연구 구조화를 통해 확인된 연구 커뮤니티 별 연구 동향 분석에 활용
<p align="center"><img src="./examples/community_year_trend.png" width="250" height="250">

## 4. keyword_evolution.html
연구 커뮤니티 별 성숙도에 따른 상위 키워드의 비율 변화 그래프. 연구 구조화를 통해 확인된 연구 커뮤니티 별 시간에 따른 상위 키워드의 빈도 변화를 평가
<p align="center"><img src="./examples/keyword_evolution.png" width="250" height="250">
