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
        # Load KARS class
        self.KARS_class = KARS()

        # Load DB interface
        self.load_DB_interface = gr.Interface(
            self.load_DB,
            [
                gr.Textbox(lines=1, label="DB name (write name of DB constructed by KBSE)")
            ],
            [
                # inhibit add column and rows
                gr.Dataframe(col_count=1, col_label="DB List", label="DB List", interactive=False),
                gr.Textbox(lines=1, label="DB status")
            ]
        )

        # construct PSPP network interface
        self.construct_PSPP_network = gr.Interface(
            self.construct_PSPP_network,
            [
            ],
            [
                # print run-time status
                gr.Textbox(lines=1, label="PSPP network status")
            ]
        )

        # research trend analysis interface 
        self.research_trend_analysis = gr.Interface(
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
                self.construct_PSPP_network,
                self.research_trend_analysis
            ],
            [
                "login_session",
                "load_DB",
                "collect_metadata",
                "construct_PSPP_network",
                "research_trend_analysis"
            ]
        ).queue(concurrency_count=1).launch(share=True)

    def load_DB(self, DB_name):
        self.DB_name = DB_name
        # 만일 self.DB_list 가 존재하지 않는다면,
        if not hasattr(self, "DB_list"):
            self.DB_list = [[]]
        if self.User == "":
            return self.DB_list, "Please login first"
        
        if DB_name == "":
            # 현존하는 DB Loading
            self.DB_list = [[DB] for DB in os.listdir(f"")]
            return self.DB_list, "DB list is loaded"
        elif DB_name in os.listdir(f"{self.DB_path}/{self.User}"):
            self.DB_list = [[DB] for DB in os.listdir(f"")]
            return self.DB_list, f"{DB_name} is loaded"
        else:
            os.makedirs(f"{self.DB_path}/{self.User}/{self.DB_name}")
            self.DB_list = [[DB] for DB in os.listdir(f"")]
            return self.DB_list, f"{DB_name} is created"

    def construct_PSPP_network(self, progress=gr.Progress()):
        # progress
        progress(0, desc="Please wait for a while...")

        # 터미널 출력을 저장할 파일의 경로를 지정합니다.
        log_filename = f"{self.DB_path}/{self.User}/{self.DB_name}/construct_PSPP_network.log"

        # 이전의 sys.stdout을 저장해둡니다.
        original_stdout = sys.stdout

        # Logger 클래스의 인스턴스를 생성하여 sys.stdout을 변경합니다.
        sys.stdout = Logger(log_filename)

        # collect metadata
        DB_path = self.DB_path + "/" + self.User + "/" + self.DB_name
        self.KARS_class.construct_PSPP_network(DB_path)

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
        log_filename = f"{self.DB_path}/{self.User}/{self.DB_name}/research_trend_analysis.log"

        # 이전의 sys.stdout을 저장해둡니다.
        original_stdout = sys.stdout

        # Logger 클래스의 인스턴스를 생성하여 sys.stdout을 변경합니다.
        sys.stdout = Logger(log_filename)

        # collect metadata
        DB_path = self.DB_path + "/" + self.User + "/" + self.DB_name
        DB_name = self.DB_name
        research_maturity_plot, community_year_trend_plot, keyword_evolution_plot = self.KARS_class.research_trend_analysis(DB_path, DB_name, keyword_limit, weight_limit, min_year, start_PLC, end_PLC, top_rank)

        # 원래의 sys.stdout으로 돌아갑니다.
        sys.stdout = original_stdout

        # read log file
        with open(log_filename, "r") as f:
            log = f.read()

        return log, research_maturity_plot, community_year_trend_plot, keyword_evolution_plot
    
if __name__ == "__main__":
    KARS_GUI_class = KARS_GUI()

