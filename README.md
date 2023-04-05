# Keyword-Based Research Structurization (KBRS)
키워드에 기반해 논문의 메타데이터를 자동으로 수집, 연구 구조화, 연구 트랜드 분석을 자동화합니다.

# Installation
## 1. anaconda 설치
    conda create -n KBRS python==3.9
    conda activate KBRS

## 2. python libraries 설치
    pip install pandas
    pip install tqdm
    pip install crossrefapi
    pip install -U pip setuptools wheel
    pip install scikit-learn
    pip install jsonlines
    pip install pylatexenc
    pip install networkx[default]
    pip install tabulate
    pip install -U spacy
    pip install torch==1.12.1+cu102  --extra-index-url https://download.pytorch.org/whl/cu102
    pip install gradio

## 3. Spacy 언어팩 설치
    python -m spacy download en_core_web_sm
    python -m spacy download en_core_web_md

# Usage
    KARS_GUI.py 실행 후, https://127.0.0.1:7860 에 접속해 관련 설정 후 제출

# Results
## 1. n (xx.xx%) 폴더들
    연구 분야를 구조화한 결과로, 각 커뮤니티별로 구분되어 있습니다.
    1) n: 구분된 커뮤니티의 index
    2) (xx.xx%): 전체 키워드의 Pagerank 대비 커뮤니티의 키워드 Pagerank 비율

    폴더 내에 존재하는 파일은 다음과 같습니다.
    1) community.html: 구조화된 커뮤니티의 키워드 interactive graph
    2) pagerank.csv: 구조화된 커뮤니티의 주요 키워드를 1순위부터 기술
    3) pagerank_doc.csv: 구조화된 커뮤니티의 주요 키워드를 포함하는 논문 정보를 1순위부터 기술
    4) subgraph.json: 구조화된 커뮤니티의 그래프 데이터

    또한, 해당 커뮤니티의 키워드 분포 비율이 >20% 인 경우, 커뮤니티를 한번 더 세분화 합니다.

## 2. total_year_trend.png
    전체 커뮤니티의 연도에 따른 키워드 수의 변화입니다.
    해당 연구 분야의 전체적인 연구 트렌드 현황을 분석하는데 사용됩니다.

## 3. community_year_trend.png
    각 커뮤니티 별 연도에 따른 키워드 분포의 변화입니다.
    해당 연구 분야의 특정 커뮤니티의 연구 트렌드 현황 분석에 사용됩니다.

## 4. gaussian_interpolation.png
    전체 연구 트렌드를 Product Life Cycle (PLC) 이론에 기반해 가우시안 추정한 결과입니다.
    그림내에서 점선은 각각 다음을 의미합니다.
    1) 첫번쨰 점선: Development 단계 - Introduction 단계. mu-3sigma
    2) 두번째 점선: Introduction 단계 - Growth 단계. mu-2sigma
    3) 세번째 점선: Growth 단계 - Maturity 단계. mu
    4) 네번째 점선: Maturity 단계 - Decline 단계. mu+sigma