import os
from KBSE.KBSE import KBSE
from network.PSPP_network import PSPP_network
from analysis.research_trend import research_trend
from analysis.PSPP_relation import PSPP_relation

class KARS:
    def __init__(self):
        pass
    
    def collect_metadata(self, DB_path, engine_list, keyword_list):
        KBSE_class = KBSE(engine_list, DB_path)
        for keyword in keyword_list:
            KBSE_class.unquery_keyword(keyword)
            KBSE_class.collect_metadata_by_keyword(keyword)
            KBSE_class.integrate_metadata(check_open_access=False)
            KBSE_class.preprocess_metadata()
            KBSE_class.save_metadata()
        KBSE_class.document_classification()

    def construct_PSPP_network(self, DB_path):
        PSPP_network_class = PSPP_network(DB_path)
        PSPP_network_class.PSPP_co_occurrence()
        PSPP_network_class.PSPP_relationship(file_type="KBSE", para_type="abstract")
        PSPP_network_class.construct_PSPP_network(edge_type="co_occurrence", para_type="title")
        PSPP_network_class.construct_PSPP_network(edge_type="relationship", para_type="abstract")

    def research_trend_analysis(self, DB_path, DB_name, keyword_limit=80, weight_limit=2, min_year=None, start_PLC='development', end_PLC='maturity', top_rank=30):
        research_trend_class = research_trend(DB_name, DB_path)
        research_trend_class.keyword_selection(keyword_limit=80)
        research_trend_class.community_detection(weight_limit=2)
        research_maturity_plot = research_trend_class.research_maturity(min_year=min_year)
        community_year_trend_plot = research_trend_class.community_year_trend(start_PLC=start_PLC)
        keyword_evolution_plot = research_trend_class.keyword_evolution(top_rank=30, start_PLC=start_PLC, end_PLC=end_PLC)
        return research_maturity_plot, community_year_trend_plot, keyword_evolution_plot

    def problem_analysis(self):
        pass

    def materials_trend_analysis(self):
        pass

    def PSPP_relation_analysis(self, DB_path, node_weight_limit=10, edge_weight_limit=5):
        PSPP_relation_class = PSPP_relation(self.DB_path, para_type="abstract")
        PSPP_relation_class.construct_PSPP_tree(para_type="abstract", node_weight_limit=node_weight_limit, edge_weight_limit=edge_weight_limit)
        #PSPP_relation_class.community_detection()
        #PSPP_relation_class.search_keyword()
        PSPP_relation_class.search_path(edge_weight_limit=10, target_node="speed", except_node_list=['device'])
        #PSPP_relation_class.search_path_by_paper(source_node="reliability", target_node="endurance")
