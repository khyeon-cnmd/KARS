import os
import gradio as gr
import pandas as pd
import shutil
import time
import jsonlines
import datetime
from collect.crossref import crossref
from extract.keyword_extract import keyword_extract
from network.research_structure import research_structure
from analysis.trend_analysis import trend_analysis
from analysis.Relationship_analysis import relationship_analysis

class KARS:
    def __init__(self, path):
        self.path = path
        self.email = ""
        self.DB_name = ""
        self.user_name = ""
        pass

    def US(self, email, DB_name):
        # 0. Initialize
        self.email = email
        self.DB_name = DB_name

        # 1. Get user name from email
        self.user_name = email.split("@")[0]

        # 2. Save path
        self.save_path = f"{self.path}/results/{self.user_name}/{self.DB_name}"

        # 3. Make directory for saving data
        past_data = {"Article_collection": "X", "Keyword_extraction": "X", "Research_structure": "X", "Trend_analysis": "X"}
        if not os.path.exists(f"{self.save_path}"):
            os.makedirs(f"{self.save_path}")
        else:
            if os.path.exists(f"{self.save_path}/Article_collection"):
                past_data["Article_collection"] = "O"
            if os.path.exists(f"{self.save_path}/Keyword_extraction"):
                past_data["Keyword_extraction"] = "O"
            if os.path.exists(f"{self.save_path}/Research_structure"):
                past_data["Research_structure"] = "O"
            if os.path.exists(f"{self.save_path}/Trend_analysis"):
                past_data["Trend_analysis"] = "O"
                
        # 4. Return the result
        df = pd.DataFrame(past_data, index=[0])

        # 5. Save the inputs
        jsonlines.open(f"{self.save_path}/inputs.json", mode='w').write({"email": email, "DB_name": DB_name})


        return df

    def AC(self, search_engines, search_keywords, progress=gr.Progress()):
        # 1. Make directory for saving data
        if not os.path.exists(f"{self.save_path}/Article_collection"):
            os.makedirs(f"{self.save_path}/Article_collection")

        # 2. make search keyword list
        keywords = search_keywords.split(",")
        
        # 3. Collect articles from Crossref
        progress(0, desc="Collecting articles from Crossref")
        time.sleep(1)
        if "crossref" in search_engines:
            crossref(keywords, self.save_path, self.DB_name, self.email) 

        # 4. Return the result
        df = pd.read_csv(f"{self.save_path}/Article_collection/{self.DB_name}.csv")

        # 5. Save the inputs
        jsonlines.open(f"{self.save_path}/Article_collection/inputs.json", mode='w').write({"search_engines": search_engines, "search_keywords": search_keywords})

        return df

    def KE(self, tokenizer_type, progress=gr.Progress()):
        # 1. Make directory for saving data
        if not os.path.exists(f"{self.save_path}/Keyword_extraction"):
            os.makedirs(f"{self.save_path}/Keyword_extraction")

        # 2. keyword extract settings
        
        # 3. Extract keywords from articles
        progress(0, desc="Extracting keywords from articles")
        time.sleep(1)
        keyword_extract(self.save_path, self.DB_name, tokenizer_type)

        # 4. Return the result
        df = pd.read_csv(f"{self.save_path}/Keyword_extraction/{self.DB_name}_cleaned.csv")

        # 5. Save the inputs
        jsonlines.open(f"{self.save_path}/Keyword_extraction/inputs.json", mode='w').write({"tokenizer_type": tokenizer_type})

        return df

    def RS(self, text_type, edge_type, start_year, end_year,  modular_algorithm, modularity_resolution, modularity_seed, pagerank_filter, save_docs, progress=gr.Progress()):
        # 1. Make directory for saving data
        if not os.path.exists(f"{self.save_path}/Research_structure"):
            os.makedirs(f"{self.save_path}/Research_structure")

        # 2. Research structure settings
        text_type = text_type
        edge_type = edge_type
        if start_year == "":
            start_year = 1500
        start_year = int(start_year)
        if end_year == "":
            end_year = datetime.datetime.now().year
        end_year = int(end_year)
        modular_algorithm = modular_algorithm
        modularity_resolution = float(modularity_resolution)
        modularity_seed = int(modularity_seed)
        pagerank_filter = int(pagerank_filter)
        if save_docs == "Yes":
            save_docs = True
        else:
            save_docs = False

        # 5. Run community detection
        progress(0, desc="Running community detection")
        time.sleep(1)
        research_structure(self.save_path, self.DB_name, text_type, edge_type, start_year, end_year, modular_algorithm, modularity_resolution, modularity_seed, pagerank_filter, save_docs)

        # 6. Return the result
        integrated_pagerank = f"{self.save_path}/Research_structure/integrated_pagerank.png"
        df = pd.DataFrame()
        for root, dirs, files in os.walk(f"{self.save_path}/Research_structure"):
            for dir in dirs:
                pagerank = pd.read_csv(f"{self.save_path}/Research_structure/{dir}/pagerank.csv")
                df[dir] = pagerank["node"] + "(" + pagerank["pagerank"].astype(str) + ")"

        # 5. Save the inputs
        jsonlines.open(f"{self.save_path}/Research_structure/inputs.json", mode='w').write({"text_type": text_type, "edge_type": edge_type, "modular_algorithm": modular_algorithm, "modularity_resolution": modularity_resolution, "modularity_seed": modularity_seed, "pagerank_filter": pagerank_filter})

        return integrated_pagerank, df

    def TA(self, start_year, end_year, community_limit, fitting_method, progress=gr.Progress()):
        # 1. Make directory for saving data
        if not os.path.exists(f"{self.save_path}/Trend_analysis"):
            os.makedirs(f"{self.save_path}/Trend_analysis")

        # 2. Settings
        if end_year == "":
            end_year = datetime.datetime.now().year
        end_year = int(end_year)
        if start_year == "":
            start_year = end_year - 50
        start_year = int(start_year)
        community_limit = float(community_limit)
        fitting_method = fitting_method

        # 4. Run trend analysis
        progress(0, desc="Analyzing research trend")
        time.sleep(1)
        trend_analysis(self.save_path, start_year, end_year, fitting_method, community_limit)

        # 5. Return the result
        gaussian_interpolation = f"{self.save_path}/Trend_analysis/gaussian_interpolation.png"
        total_year_trend = f"{self.save_path}/Trend_analysis/total_year_trend.png"
        community_year_trend = f"{self.save_path}/Trend_analysis/community_year_trend.png"    
        shutil.make_archive(f"{self.save_path}", 'zip', f"{self.save_path}")
        while 1:
            if os.path.exists(f"{self.save_path}.zip"):
                break
            else:
                time.sleep(1)
        zip = f"{self.save_path}.zip"

        # 6. Save the inputs
        jsonlines.open(f"{self.save_path}/Trend_analysis/inputs.json", mode='w').write({"start_year": start_year, "end_year": end_year, "community_limit": community_limit, "fitting_method": fitting_method})

        return gaussian_interpolation, total_year_trend, community_year_trend, zip

    def RA(self, progress=gr.Progress()):
        # 1. Make directory for saving data
        if not os.path.exists(f"{self.save_path}/Relationship_analysis"):
            os.makedirs(f"{self.save_path}/Relationship_analysis")

        # 2. Settings

        # 4. Run trend analysis
        progress(0, desc="Analyzing research trend")
        time.sleep(1)
        self.RA_model = relationship_analysis(self.save_path, self.DB_name)

        # 5. Return the result
        df = pd.read_csv(f"{self.save_path}/Relationship_analysis/keyword_list.csv")

        return df

    def RA2(self, material, text_type, edge_type, progress=gr.Progress()):
        # 1. Make directory for saving data
        if not os.path.exists(f"{self.save_path}/Relationship_analysis"):
            os.makedirs(f"{self.save_path}/Relationship_analysis")

        # 2. Research structure settings
        text_type = text_type
        edge_type = edge_type
            
        # 4. Run trend analysis
        progress(0, desc="Analyzing research trend")
        time.sleep(1)
        self.RA_model(material, text_type, edge_type)    

        return self.RA_model.info

    def RA3(self, source, target, internodes, NER, max_distance, progress=gr.Progress()):
        # 1. Make directory for saving data
        if not os.path.exists(f"{self.save_path}/Relationship_analysis"):
            os.makedirs(f"{self.save_path}/Relationship_analysis")

        # 2. settings
        source = source
        target = target
        internodes = internodes.replace(" ", "").split(",")
        if internodes == [""]:
            internodes = []
        if "" in NER:
            NER.remove("")
        NER = NER
        max_distance = int(max_distance)
        

        # 4. Run trend analysis
        progress(0, desc="Analyzing research trend")
        time.sleep(1)
        path = self.RA_model.path_find(source, target, internodes, NER, max_distance)

        path_str = ""
        for p in path:
            path_str += f"{p}\n"
        
        return path_str      

if __name__ == "__main__":
    # 0. Get the operating path
    path = os.path.dirname(os.path.abspath(__file__))

    # 1. KARS operate
    kars = KARS(path)

    # 2. User setting
    input_list = []
    for user in os.listdir(f"{path}/results"):
        for DB in os.listdir(f"{path}/results/{user}"):
            if os.path.exists(f"{path}/results/{user}/{DB}/inputs.json"):
                dict = jsonlines.open(f"{path}/results/{user}/{DB}/inputs.json").read()
                input = [dict["email"], dict["DB_name"]]
                input_list.append(input)

    user_setting = gr.Interface(
        kars.US,
        [
            gr.Textbox(label="E-mail",info="Your email address",lines=1,value="example@email.com"),
            gr.Textbox(label="DB name",info="research field",lines=1,value=""),
        ],
        [
            gr.Dataframe(row_count=(1,"fixed"),col_count=(4,"fixed"), interactive=False,info="Identified exisiting data"),
        ],
        input_list
    )

    # 3. Article collection DB
    AC_list = []
    if not kars.user_name == "" and not kars.DB_name == "":
        if os.path.exists(f"{path}/results/{kars.user_name}/{kars.DB_name}/Article_collection/inputs.json"):
                dict = jsonlines.open(f"{path}/results/{kars.user_name}/{kars.DB_name}/Article_collection/inputs.json").read()
                input = [dict["email"], dict["DB_name"]]
                AC_list.append(input)

    article_collection = gr.Interface(
        kars.AC,
        [
            gr.CheckboxGroup(label=["crossref"], info="Check the database to receive bibliographic data", value=["crossref"]),
            gr.Textbox(label="search keywords",info="Please write the search keywords about the research field",lines=3,value="ReRAM, Conductive Filament, Memristor -PCRAM"),
        ],
        [
            gr.Dataframe(row_count=(20,"dynamic"),col_count=(5,"dynamic"), interactive=False,info="Collected articles"),
        ],
        AC_list
    )

    keyword_extraction = gr.Interface(
        kars.KE,
        [
            gr.Radio(["efficiency", "accurate"], label="Tokenizer type", info="Choose between efficiency(recommand) and accurate"),
        ],
        [
            gr.Dataframe(row_count=(20,"dynamic"),col_count=(5,"dynamic"), interactive=False,info="Keyword extracted articles"),
        ],
        
    )

    research_structuring = gr.Interface(
        kars.RS,
        [            
            gr.Radio(["title", "abstract"], label="Text type", info="Choose the data source between title(recommand) and abstract", value="title"),
            gr.Radio(["co-occurrence", "neighbor"], label="Edge type", info="Choose the edge type between co-occurrence(recommand) and neighbor", value="co-occurrence"),
            gr.Textbox(label="Start year",info="Set the start year to figure out the research structure(ex. 1980, Blank = all)",lines=1,value=""),
            gr.Textbox(label="End year",info="Set the end year to figure out the research structure(ex. 2023, Blank = all)",lines=1,value=""),
            gr.Radio(["louvain","greedy","girvan-newman"], label="Modular algorithm", info="Choose the algorithm for community detection between louvain(recommand), greedy and Girvan-newman", value="louvain"),
            gr.Slider(0,2,value=1,label="Modularity resolution",info="Choose the modularity resolution between 0(more community) and 2(less community)"),
            gr.Textbox(label="Modularity seed",info="Set the values for reproducible results",lines=1,value="42"),
            gr.Textbox(label="Pagerank filter",info="Filter keywords from top to n% sorted by pagerank",lines=1,value="70"),
            gr.Radio(["Yes", "No"],label="Save docs", info="Do we use document classification?", value="No"),
        ],
        [
            gr.Image(type="pil",info="The result of integrated pagerank"),
            gr.Dataframe(row_count=(20,"fixed"),col_count=(5,"dynamic"), interactive=False,info="The top 20 keywords of structured research communities"),
        ],
    )

    trend_analyze = gr.Interface(
        kars.TA,
        [
            gr.Textbox(label="Start year",info="Set the start year to figure out the research structure(ex. 1980, Blank = all)",lines=1,value=""),
            gr.Textbox(label="End year",info="Set the end year to figure out the research structure(ex. 2023, Blank = all)",lines=1,value=""),
            gr.Textbox(label="Community limit",info="Set the percentage limit of community not to visualize",lines=1,value="1"),
            gr.Radio(["gaussian"], label="Fitting method", info="Choose the fitting method between gaussian", value="gaussian"),
        ],
        [
            gr.Image(type="pil",info="The result of gaussian interpolation"),
            gr.Image(type="pil",info="The result of total year trend"),
            gr.Image(type="pil",info="The result of community year trend"),
            gr.File(type="file",info="The result file of the research trend analysis"),
        ],
    )

    relationship_analyze = gr.Interface(
        kars.RA,
        [
        ],
        [
            gr.DataFrame(type="pandas",info="The materials list from the graph"),
        ],
    )

    relationship_analyze2 = gr.Interface(
        kars.RA2,
        [
            gr.Textbox(label="Material",info="Please write the material name",lines=1,value=""),
            gr.Radio(["title", "abstract"], label="Text type", info="Choose the data source between title(recommand) and abstract", value="title"),
            gr.Radio(["co-occurrence", "neighbor"], label="Edge type", info="Choose the edge type between co-occurrence(recommand) and neighbor", value="co-occurrence"),
        ],
        [
            gr.Textbox(label="graph_info", info="The result of the graph",lines=5,value=""),
        ]
    )

    relationship_analyze3 = gr.Interface(
        kars.RA3,
        [
            gr.Textbox(label="Source",info="Please write the source node",lines=1,value=""),
            gr.Textbox(label="Target",info="Please write the target node",lines=1,value=""),
            gr.Textbox(label="Internodes",info="Please write the internodes using ,",lines=1,value=""),
            gr.CheckboxGroup(["material", "value"],label="NER filter", info="filter the path having NER value", value=[""]),
            gr.Textbox(label="Max distance",info="Please write the max distance",lines=1,value="4"),
        ],
        [
            gr.Textbox(label="Result",info="The result of the shortest path",lines=5,value="")
        ]
    )


    gr.TabbedInterface([user_setting, article_collection, keyword_extraction, research_structuring, trend_analyze, relationship_analyze, relationship_analyze2, relationship_analyze3], ["User setting","Article collection", "Keyword extraction", "Research structuring", "Trend analysis", "Relationship analysis", "Relationship analysis2","Relationship analysis3",]).queue(concurrency_count=1).launch(share=True)