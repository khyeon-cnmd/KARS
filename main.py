import os
from collect.crossref import crossref
from extract.keyword_extract import keyword_extract
from network.research_structure import research_structure
from analysis.trend_analysis import trend_analysis

# 1. Config
email = "khyeon@postech.ac.kr" #<- your email

# 2. keyword list
keywords = [
    "RRAM",
    "ReRAM",
    "Oxram",
    "OxrRAM",
    "cbram",
    "electrochemical metallization memory",
    "valence change memory",
    "resistive switching",
    "filament switching",
    "conductive filament",
    "conductive bridge",
    "oxygen vacancies filament"
    ]

# 3. save_path
save_path = "/home1/khyeon/Researches/2_Text_mining/KBRS"
DB_name = "DSSC"
mode = "efficiency"
text_type = "abstract" #"title" #"abstractr" <- not recommanded until collected more data
edge_count_type ="neighbor" #"co-occurrence"  
ngram_range = (1,1) #(1,2) <- not recommanded until collected more data
community_resolution = 1 # small value -> larger community size
community_seed = 42 # None <- for random seed
filter_percent = 30 # Filter keywords from top to n% sorted by pagerank\
fit_type = "gu"
community_limit = 1 # limit of percentage to plot community_year_trend
year_range = 50 # analysis trend for the last 50 years


if __name__ == "__main__":
    if not os.path.exists(f"{save_path}/{DB_name}"):
        os.makedirs(f"{save_path}/{DB_name}")

    # 1. data collection from crossref
    #crossref(keywords,save_path, DB_name, email) #works!

    # 2. keyword to Graph data
    #keyword_extract(save_path, DB_name, mode, text_type, edge_count_type, ngram_range)

    # 3. Research field structurization
    #graph_network(save_path, DB_name, text_type, filter_percent, community_seed, community_resolution)

    # 4. Research field trend analysis
    trend_analysis(f"{save_path}/{DB_name}", fit_type, community_limit, year_range)