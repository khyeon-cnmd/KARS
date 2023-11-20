import os
import sys
import gradio as gr
from KARS import KARS

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

class KARS_GUI:
    def __init__(self):
        # Load DB interface
        self.load_DB_interface = gr.Interface(
            self.load_DB,
            [
                gr.Textbox(lines=1, label="DB path (write path of DB constructed by KBSE)")
            ],
            [
                # inhibit add column and rows
                gr.Dataframe(col_count=1, col_label="DB List", label="DB List", interactive=False),
                gr.Textbox(lines=1, label="DB status")
            ]
        )

        # keyword_extraction interface
        self.keyword_extraction_interface = gr.Interface(
            self.keyword_extraction,
            [
                gr.Radio(["efficiency", "accuracy"], label="UPOS_model", value="efficient", description="The model for UPOS tagging (efficiency: en_core_web_sm, accuracy: en_core_web_trf)"),
            ],
            [
                # print run-time status
                gr.Textbox(lines=1, label="keyword extraction status")
            ]
        )

        # network_construction interface
        self.network_construction_interface = gr.Interface(
            self.network_construction,
            [
            ],
            [
                # print run-time status
                gr.Textbox(lines=1, label="network construction status")
            ]
        )

        # research trend analysis interface 
        self.research_trend_analysis_interface = gr.Interface(
            self.research_trend_analysis,
            [
                gr.Number(label="keyword limit", value=80, step=1, min_value=1, max_value=100, description="The % of keywords to be selected"),
                gr.Number(label="weight limit", value=2, step=1, min_value=1, max_value=100, description="The weight % criteria for communities"),
                gr.Radio(["development", "introduction", "growth", "maturity", "decline"], label="start PLC", value="introduction", description="The start PLC of research trend analysis"),
                gr.Radio(["development", "introduction", "growth", "maturity", "decline"], label="end PLC", value="maturity", description="The end PLC of research trend analysis"),
                gr.Number(label="top rank", value=20, step=1, min_value=1, max_value=100, description="The top rank of keyword evolution"),
            ],
            [
                # print run-time status
                gr.Textbox(lines=1, label="research trend analysis status"),
                gr.Plot(label="research maturity"),
                gr.Plot(label="community year trend"),
                gr.Plot(label="keyword evolution")
            ]
        )

        # Tabbed interface
        self.tab_interface = gr.TabbedInterface(
            [
                self.load_DB_interface,
                self.keyword_extraction_interface,
                self.network_construction_interface,
                self.research_trend_analysis_interface
            ],
            [
                "load_DB",
                "keyword_extraction",
                "network_construction",
                "research_trend_analysis"
            ]
        ).queue(concurrency_count=1).launch(share=True)

    def load_DB(self, DB_path):
        self.DB_path = DB_path

        # 만일 self.DB_path 가 존재하지 않는다면,
        if os.path.isdir(self.DB_path) == False:
            return [], "Please check your DB path"
        else:
            # Load KARS class
            self.KARS_class = KARS(self.DB_path)
        
            # DB_path 안에 있는 모든 폴더를 DB_list에 저장합니다.
            Folder_list = [[DB] for DB in os.listdir(self.DB_path)]
            return Folder_list, "DB is loaded"

    def keyword_extraction(self, UPOS_model, progress=gr.Progress()):
       # progress
        progress(0, desc="Please wait for a while...")

        # 터미널 출력을 저장할 파일의 경로를 지정합니다.
        log_filename = f"{self.DB_path}/KARS/keyword_extraction.log"

        # 이전의 sys.stdout을 저장해둡니다.
        original_stdout = sys.stdout

        # Logger 클래스의 인스턴스를 생성하여 sys.stdout을 변경합니다.
        sys.stdout = Logger(log_filename)

        # keyword_extraction
        self.KARS_class.keyword_extraction(UPOS_model)

        # 원래의 sys.stdout으로 돌아갑니다.
        sys.stdout = original_stdout

        # read log file
        with open(log_filename, "r") as f:
            log = f.read()

        return log

    def network_construction(self, progress=gr.Progress()):
        # progress
        progress(0, desc="Please wait for a while...")

        # 터미널 출력을 저장할 파일의 경로를 지정합니다.
        log_filename = f"{self.DB_path}/KARS/network_construction.log"

        # 이전의 sys.stdout을 저장해둡니다.
        original_stdout = sys.stdout

        # Logger 클래스의 인스턴스를 생성하여 sys.stdout을 변경합니다.
        sys.stdout = Logger(log_filename)

        # network construction
        self.KARS_class.network_construction()

        # 원래의 sys.stdout으로 돌아갑니다.
        sys.stdout = original_stdout

        # read log file
        with open(log_filename, "r") as f:
            log = f.read()

        return log

    def research_trend_analysis(self, keyword_limit, weight_limit, start_PLC, end_PLC, top_rank, progress=gr.Progress()):
        min_year=None
        keyword_limit = int(keyword_limit)
        weight_limit = int(weight_limit)
        top_rank = int(top_rank)
        
        # progress
        progress(0, desc="Please wait for a while...")

        # 터미널 출력을 저장할 파일의 경로를 지정합니다.
        log_filename = f"{self.DB_path}/KARS/research_trend_analysis.log"

        # 이전의 sys.stdout을 저장해둡니다.
        original_stdout = sys.stdout

        # Logger 클래스의 인스턴스를 생성하여 sys.stdout을 변경합니다.
        sys.stdout = Logger(log_filename)

        # collect metadata
        research_maturity_plot, community_year_trend_plot, keyword_evolution_plot = self.KARS_class.research_trend_analysis(keyword_limit, weight_limit, min_year, start_PLC, end_PLC, top_rank)

        # 원래의 sys.stdout으로 돌아갑니다.
        sys.stdout = original_stdout

        # read log file
        with open(log_filename, "r") as f:
            log = f.read()

        return log, research_maturity_plot, community_year_trend_plot, keyword_evolution_plot
    
if __name__ == "__main__":
    KARS_GUI_class = KARS_GUI()

