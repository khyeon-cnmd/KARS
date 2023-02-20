import os
from collect.crossref import crossref
from extract.keyword_extract import keyword_extract
from network.graph_network import graph_network
from analysis.trend_analysis import trend_analysis

# 1. Config
email = "khyeon@postech.ac.kr" #<- your email

# 2. keyword list
keywords = [
    "HEA",
    #"High entropy alloy",
    ]

# 3. save_path
save_path = "/home1/khyeon/Researches/2_Text_mining/KBRS"
DB_name = "Organic"
mode = "efficiency"
text_type = "abstract"
fit_type = "gu"

if __name__ == "__main__":
    if not os.path.exists(f"{save_path}/{DB_name}"):
        os.makedirs(f"{save_path}/{DB_name}")

    # 1. data collection from crossref
    crossref(keywords,save_path, DB_name, email) #works!

    # 2. keyword to Graph data
    keyword_extract(save_path, DB_name, mode, text_type)

    # 3. Research field structurization
    gn = graph_network(save_path, DB_name)
    gn.save_graph()
    for path, dirs, file in os.walk(f"{save_path}/{DB_name}"):
        for dir in dirs:
            share_percent = dir.split("%")[0].split("(")[1]
            if float(share_percent) > 20:
                gn(save_path=f"{path}/{dir}")
                gn.save_graph()

    # 4. Research trend analysis
    ta = trend_analysis(save_path=f"{save_path}/{DB_name}", fit_type=fit_type)
    for path, dirs, file in os.walk(f"{save_path}/{DB_name}"):
        for dir in dirs:            
            share_percent = dir.split("%")[0].split("(")[1]
            if float(share_percent) > 20:
                ta(save_path=f"{path}/{dir}", fit_type=fit_type)