import os
from extract.keyword_extract import keyword_extraction
from network.network_construct import network_construction
from analysis.research_trend import research_trend

class KARS:
    def __init__(self, DB_path):
        self.DB_path = DB_path
        if os.path.isdir(self.DB_path + "/KARS") == False:
            os.mkdir(self.DB_path + "/KARS")

    def keyword_extraction(self,UPOS_model):
        keyword_extraction_class = keyword_extraction(self.DB_path, UPOS_model=UPOS_model)
        keyword_extraction_class.keyword_tokenization()

    def network_construction(self):
        network_construction_class = network_construction(self.DB_path)
        network_construction_class.network_construct()
        network_construction_class.network_integrate()

    def research_trend_analysis(self, keyword_limit=80, weight_limit=2, min_year=None, start_PLC='development', end_PLC='maturity', top_rank=30):
        research_trend_class = research_trend(self.DB_path)
        research_trend_class.keyword_selection(keyword_limit=keyword_limit)
        research_trend_class.community_detection(weight_limit=weight_limit)
        research_maturity_plot = research_trend_class.research_maturity(min_year=min_year)
        community_year_trend_plot = research_trend_class.community_year_trend(start_PLC=start_PLC)
        keyword_evolution_plot = research_trend_class.keyword_evolution(top_rank=top_rank, start_PLC=start_PLC, end_PLC=end_PLC)

        return research_maturity_plot, community_year_trend_plot, keyword_evolution_plot
