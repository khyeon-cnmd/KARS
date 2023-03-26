import os
from collect.crossref import crossref
from extract.keyword_extract import keyword_extract
from network.graph_network import graph_network
from analysis.trend_analysis import trend_analysis
from analysis.MPSPP_analysis import MPSPP_analysis

# 1. Config
email = "khyeon@postech.ac.kr" #<- your email

# 2. keyword list
keywords = [
    "Perovskite solar cell",
    "PSCs"
    ]

# 3. save_path
save_path = "/home1/khyeon/Researches/2_Text_mining/KBRS"
DB_name = "ReRAM"
mode = "efficiency"
text_type = "title" #"abstract" <- not recommanded until collected more data
ngram_range = (1,1) #(1,2) <- not recommanded until collected more data
community_resolution = 1 # small value -> larger community size
community_seed = 42 # None <- for random seed
filter_percent = 70
fit_type = "gu" #"gaussian" <- scipy based fitting
subgraph_limit = 20 # limit of percentage to divede the subgraph
community_limit = 1 # limit of percentage to plot community_year_trend
year_range = 50 # analysis trend for the last 50 years

if __name__ == "__main__":
    if not os.path.exists(f"{save_path}/{DB_name}"):
        os.makedirs(f"{save_path}/{DB_name}")

    # 1. data collection from crossref
    #crossref(keywords,save_path, DB_name, email) #works!

    # 2. keyword to Graph data
    #keyword_extract(save_path, DB_name, mode, text_type, ngram_range)

    # 3. Research field structurization
    graph_network(save_path, DB_name, filter_percent, community_seed, community_resolution)

    # 4. MPSPP analysis
    MPSPP_analysis(save_path, DB_name)


    # 4. Research trend analysis
    #ta = trend_analysis(save_path=f"{save_path}/{DB_name}", fit_type=fit_type, community_limit=community_limit, year_range=year_range)
    #for path, dirs, file in os.walk(f"{save_path}/{DB_name}"):
    #    for dir in dirs:          
    #        share_percent = dir.split("%")[0].split("(")[1]
    #        if float(share_percent) > subgraph_limit:
    #            ta(save_path=f"{path}/{dir}")