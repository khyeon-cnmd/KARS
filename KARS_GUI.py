import os
import gradio as gr
import pandas as pd
import shutil
from collect.crossref import crossref
from extract.keyword_extract import keyword_extract
from network.graph_network import graph_network
from analysis.trend_analysis import trend_analysis


def KARS(email,DB_name,search_keywords,processes,keyword_extract_method,text_type,edge_count_type,modular_algorithm, modularity_resolution,modularity_seed,filter_percent,fitting_method,year_range):
    # 1. Get user name from email
    user_name = email.split("@")[0]

    # 1. Save path
    save_path = f"/home1/khyeon/Researches/KARS/results/{user_name}"

    # 2. Make directory for saving data
    if not os.path.exists(f"{save_path}/{DB_name}"):
        os.makedirs(f"{save_path}/{DB_name}")
    
    # 3. default settings
    ngram_range = (1,1) #(1,2) <- not recommanded until collected more data
    community_limit = 1 # limit of percentage to plot community_year_trend

    # 4. make search keyword list
    keywords = search_keywords.split(",")

    # 5. preprocessing the varaibles
    modularity_resolution = float(modularity_resolution)
    modularity_seed = int(modularity_seed)
    filter_percent = int(filter_percent)

    # 5. Keyword based Automatic Research Structurization
    if processes <= 1:
        crossref(keywords,save_path, DB_name, email) 
    if processes <= 2:
        keyword_extract(save_path, DB_name, keyword_extract_method, text_type, edge_count_type, ngram_range)
    if processes <= 3:
        graph_network(save_path, DB_name, text_type, modular_algorithm, filter_percent, modularity_seed, modularity_resolution)
    if processes <= 4:
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
    shutil.make_archive(f"{save_path}/{DB_name}", 'zip', f"{save_path}/{DB_name}")
    zip = f"{save_path}/{DB_name}.zip"

    # 7. Save Inputs by text file
    inputs = [email,DB_name,search_keywords,processes,keyword_extract_method,text_type,edge_count_type,modular_algorithm, modularity_resolution,modularity_seed,filter_percent,fitting_method]
    with open(f"{save_path}/{DB_name}/inputs.txt", "w") as f:
        for input in inputs:
            f.write(f"{input}\n")

    return integrated_pagerank, gaussian_interpolation, total_year_trend, community_year_trend, df, zip


if __name__ == "__main__":
    #0. Read input lists
    input_list = []
    for root, dirs, files in os.walk("/home1/khyeon/Researches/KARS/results"):
        for file in files:
            if file == "inputs.txt":
                with open(os.path.join(root, file), "r") as f:
                    input_list.append(f.readlines())

    gui = gr.Interface(
        KARS,
        [
            gr.Textbox(label="E-mail",info="Your email address",lines=1,value="example@email.com"),
            gr.Textbox(label="DB name",info="research field",lines=1,value=""),
            gr.Textbox(label="search keywords",info="Please write the search keywords about the research field",lines=3,value="ReRAM, Conductive Filament, Memristor -PCRAM"),
            gr.Slider(1,4,step=1,value=1,label="Processes",info="Choose the processes between 1(collect-keyword extract-graph network-trend analysis) and 4(trend analysis)"),
            gr.Radio(["efficiency", "accurate"], label="Keyword extract method", info="Choose between efficiency(recommand) and accurate"),
            gr.Radio(["title", "abstract"], label="Text type", info="Choose the data source between title(recommand) and abstract"),
            gr.Radio(["co-occurrence", "neighbor"], label="Edge count type", info="Choose the edge count type between co-occurrence(recommand) and neighbor"),
            gr.Radio(["louvain","greedy","girvan-newman"], label="Modular algorithm", info="Choose the algorithm for community detection between louvain(recommand), greedy and Girvan-newman"),
            gr.Slider(0,2,value=1,label="Modularity resolution",info="Choose the modularity resolution between 0(more community) and 2(less community)"),
            gr.Textbox(label="Modularity seed",info="Set the values for reproducible results",lines=1,value="42"),
            gr.Textbox(label="Filter percent",info="Filter keywords from top to n% sorted by pagerank",lines=1,value="70"),
            gr.Radio(["gaussian"], label="Fitting method", info="Choose the fitting method between gaussian"),
            gr.Textbox(label="Year range",info="Range the year period to analyze(ex. 50 = 1973~2022)",lines=1,value="50"),
        ],
        [
            gr.Image(type="pil",info="The result of integrated pagerank"),
            gr.Image(type="pil",info="The result of gaussian interpolation"),
            gr.Image(type="pil",info="The result of total year trend"),
            gr.Image(type="pil",info="The result of community year trend"),
            gr.Dataframe(row_count=(20,"fixed"),col_count=(5,"dynamic"), interactive=False,info="The top 20 keywords of structured research communities"),
            gr.File(type="file",info="The result file of the research trend analysis"),
        ],
        input_list
    )

    gui.queue(concurrency_count=1).launch(share=True)
