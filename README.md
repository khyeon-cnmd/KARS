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
    #pip install pyvis
    #pip install bokeh

## 3. Spacy 언어팩 설치
    python -m spacy download en_core_web_sm 

# Usage
## 1. main.py 내 이메일 주소 변경
    email = "khyeon@postech.ac.kr"
    #본인의 이메일 주소 입력

## 2. main.py 분석 저장 폴더 변경
    save_path = "/home1/khyeon/Researches/2_Text_mining/KBRS"
    # 데이터 저장 위치 입력
    DB_name = "ReRAM"
    # 데이터베이스 이름 입력

## 3. 논문 검색 Keywords 입력
    keywords = [
        "ReRAM",
        "RRAM",
        "OxRAM",
        "OxRRAM",
        "CBRAM",
        "Electrochemical Metallization Memory",
        "Valence Change Memory",
        "Resistive Switching",
        "Filament Switching",
        "Conductive Filament",
        "Conductive Bridge",
        "Oxygen Vacancies Filament"
        ]
    # !! 확인하고자 하는 연구 영역을 잘 정의 하는 키워드로 구성해야합니다!

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
# 기타 기록용(설치 X)

#================================================== 아래쪽은 설치 필요 없습니다!>
conda install -c conda-forge "openjdk=11.0.9.1"
#pip install neo4j
# Neo4j 설치 및 APOC, GDS, Bloom Plugin 설치
#1. wget https://neo4j.com/artifact.php?name=neo4j-community-4.4.17-unix.tar.gz
#2. tar -xvf neo4j-community-4.4.17-unix.tar.gz
#3. cd neo4j-community-4.4.17/plugins
#4. wget https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/4.4.0.1/apoc-4.4.0.1-all.jar
#5. wget https://graphdatascience.ninja/neo4j-graph-data-science-2.3.0.zip
#6. wget https://neo4j.com/artifact.php?name=neo4j-bloom-2.6.1.zip
#7. unzip neo4j-graph-data-science-2.3.0.zip 
#8. unzip artifact.php\?name\=neo4j-bloom-2.6.1.zip 
#9. rm -r bloom-plugin-5.x-2.6.1.jar
#
## Neo4j conf 수정
#1. vi conf/neo4j.conf 
#2. dbms.security.procedures.unrestricted=algo.*,apoc.*,gds.*,bloom.*
#3. dbms.security.procedures.allowlist=apoc.load.*,bloom.*
#4. dbms.unmanaged_extension_classes=com.neo4j.bloom.server=/bloom
#5. dbms.security.http_auth_allowlist=/,/browser.*,/bloom.*
#
## Neo4j 구동
#1. cd neo4j-community-4.4.17/bin/ 
#2. ./neo4j console
#3. http://141.223.167.14:7474 접속해 neo4j 유저로 접속
#4. neo4j 유저의 비밀번호 변경 후 DB 생성