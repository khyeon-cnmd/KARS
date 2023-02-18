# anaconda 설치
conda create -n 가상환경 명 python==3.9
conda install -c conda-forge "openjdk=11.0.9.1"

# pip install
pip install pandas
pip install tqdm
pip install crossrefapi
pip install -U pip setuptools wheel
pip install scikit-learn
pip install jsonlines
pip install pylatexenc
pip install networkx[default]
pip install -U spacy
python -m spacy download en_core_web_sm



#================================================== 아래쪽은 설치 필요 없습니다!
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