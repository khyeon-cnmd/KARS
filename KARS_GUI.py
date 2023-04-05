import os
import gradio as gr
import pandas as pd
from collect.crossref import crossref
from extract.keyword_extract import keyword_extract
from network.graph_network import graph_network
from analysis.trend_analysis import trend_analysis


def KARS(email,DB_name,search_keywords,keyword_extract_method,text_type,edge_count_type,modularity_resolution,modularity_seed,filter_percent,fitting_method):
    # 1. Get user name from email
    user_name = email.split("@")[0]

    # 1. Save path
    save_path = f"/home1/khyeon/Researches/2_Text_mining/KBRS/results/{user_name}"

    # 2. Make directory for saving data
    if not os.path.exists(f"{save_path}/{DB_name}"):
        os.makedirs(f"{save_path}/{DB_name}")
    
    # 3. default settings
    ngram_range = (1,1) #(1,2) <- not recommanded until collected more data
    community_limit = 1 # limit of percentage to plot community_year_trend
    year_range = 50 # analysis trend for the last 50 years

    # 4. make search keyword list
    keywords = search_keywords.split(",")

    # 5. preprocessing the varaibles
    modularity_resolution = float(modularity_resolution)
    modularity_seed = int(modularity_seed)
    filter_percent = int(filter_percent)

    # 5. Keyword based Automatic Research Structurization
    crossref(keywords,save_path, DB_name, email) 
    keyword_extract(save_path, DB_name, keyword_extract_method, text_type, edge_count_type, ngram_range)
    graph_network(save_path, DB_name, text_type, filter_percent, modularity_seed, modularity_resolution)
    trend_analysis(f"{save_path}/{DB_name}", fitting_method, community_limit, year_range)
    
    # 6. outputs
    integrated_pagerank = f"{save_path}/{DB_name}/integrated_pagerank.png"
    gaussian_interpolation = f"{save_path}/{DB_name}/gaussian_interpolation.png"
    total_year_trend = f"{save_path}/{DB_name}/total_year_trend.png"
    community_year_trend = f"{save_path}/{DB_name}/community_year_trend.png"
    df = pd.DataFrame()
    for root, dirs, files in os.walk(f"{save_path}/{DB_name}"):
        for dir in dirs:
            pagerank = pd.read_csv(f"{save_path}/{DB_name}/{dir}/pagerank.csv")
            df[dir] = pagerank["node"] + "(" + pagerank["pagerank"].astype(str) + ")"

    return integrated_pagerank, gaussian_interpolation, total_year_trend, community_year_trend, df


if __name__ == "__main__":
    gui = gr.Interface(
        KARS,
        [
            gr.Textbox(label="E-mail",info="Your email address",lines=1,value=""),
            gr.Textbox(label="DB name",info="research field",lines=1,value=""),
            gr.Textbox(label="search keywords",info="Please write the search keywords about the research field(ex. ReRAM,Conductive Filament)",lines=3,value=""),
            gr.Radio(["efficiency", "accurate"], label="Keyword extract method", info="Choose between efficiency(recommand) and accurate"),
            gr.Radio(["title", "abstract"], label="Text type", info="Choose the data source between title(recommand) and abstract"),
            gr.Radio(["co-occurrence", "neighbor"], label="Edge count type", info="Choose the edge count type between co-occurrence(recommand) and neighbor"),
            gr.Slider(0,2,value=1,label="Modularity resolution",info="Choose the modularity resolution between 0(more community) and 2(less community)"),
            gr.Textbox(label="Modularity seed",info="Set the values for reproducible results",lines=1,value="42"),
            gr.Textbox(label="Filter percent",info="Filter keywords from top to n% sorted by pagerank",lines=1,value="70"),
            gr.Radio(["gu", "gaussian"], label="Fitting method", info="Choose the fitting method between gu(recommand) and gaussian"),
        ],
        [
            #gr.Dataframe(headers=["Index","Title","Abstract","Published date","DOI"],row_count=(10,"dynamic"),col_count=(5,"fixed"), interactive=False,info="The result of collected bibliographic data from crossref"),
            gr.Image(type="pil",info="The result of integrated pagerank"),
            gr.Image(type="pil",info="The result of gaussian interpolation"),
            gr.Image(type="pil",info="The result of total year trend"),
            gr.Image(type="pil",info="The result of community year trend"),
            gr.Dataframe(row_count=(20,"fixed"),col_count=(5,"dynamic"), interactive=False,info="The top 20 keywords of structured research communities"),
        ]
    )

    gui.queue(concurrency_count=1).launch(share=True)
