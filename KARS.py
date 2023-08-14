import os
from KBSE.KBSE import KBSE
from network.PSPP_network import PSPP_network
from analysis.research_trend import research_trend
from analysis.PSPP_relation import PSPP_relation

class KARS:
    def __init__(self, DB_name):
        self.DB_name = DB_name
        self.DB_path = os.path.join(os.getcwd(), DB_name)
        pass

    def collect_metadata(self, engine_list, keyword_list):
        KBSE_class = KBSE(engine_list, self.DB_path)    
        for keyword in keyword_list:
            KBSE_class.unquery_keyword(keyword)
            KBSE_class.collect_metadata_by_keyword(keyword)
            KBSE_class.integrate_metadata(check_open_access=False)
            KBSE_class.preprocess_metadata()
            KBSE_class.save_metadata()
        KBSE_class.document_classification()

    def construct_PSPP_network(self):
        PSPP_network_class = PSPP_network(self.DB_path)
        PSPP_network_class.PSPP_relationship()
        PSPP_network_class.PSPP_co_occurrence()
        PSPP_network_class.construct_PSPP_network(para_type="abstract")
        PSPP_network_class.construct_PSPP_network(para_type="title")

    def research_trend_analysis(self):
        research_trend_class = research_trend(self.DB_name, self.DB_path)
        research_trend_class.keyword_selection(keyword_limit=80)
        research_trend_class.community_detection(weight_limit=0.02)
        research_trend_class.research_maturity(min_year=None)
        research_trend_class.community_year_trend(start_PLC='development')
        research_trend_class.keyword_evolution(top_rank=30, start_PLC='development', end_PLC='introduction')

    def problem_analysis(self):
        pass

    def materials_trend_analysis(self):
        pass

    def PSPP_relation_analysis(self):
        PSPP_relation_class = PSPP_relation(self.DB_path)
        #PSPP_relation_class.community_detection()
        #PSPP_relation_class.search_keyword()
        PSPP_relation_class.search_path(edge_weight_limit=10, target_node="endurance")

        pass