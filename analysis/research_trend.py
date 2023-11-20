import os
import networkx as nx
import datetime
from bokeh.io import export_png
from bokeh.models import TabPanel, Tabs
from scipy.optimize import curve_fit
from bokeh.plotting import figure, output_file, save
from bokeh.models import HoverTool, Span, Label, Legend, ColumnDataSource, LabelSet
from bokeh.transform import dodge
from bokeh.palettes import Category20
import numpy as np


class research_trend:
    def __init__(self, DB_path):
        self.DB_name = DB_path.split("/")[-1]
        self.DB_path = DB_path + "/KARS"
        self.G = nx.read_gexf(os.path.join(self.DB_path, "KARS.gexf"))
        pass

    def keyword_selection(self, keyword_limit):
        print("Keyword selection started")
        # 1. calculate pagerank
        node_pagerank = nx.pagerank(self.G, alpha=0.85, max_iter=20, tol=1e-06, weight="weight", dangling=None)
        for key, value in self.G.nodes.data():
            self.G.nodes[key]['pagerank'] = node_pagerank[key]
        
        # 2. select keyword by keyword limit
        # 2.1. sort nodes in self.G by pagerank
        keyword_list = sorted(self.G.nodes.data(), key=lambda x: x[1]['pagerank'], reverse=True)
        weight_total = sum([self.G.nodes[key]['weight'] for key, value in self.G.nodes.data()])
        weight_sum = 0
        for keyword in keyword_list:
            keyword_weight = keyword[1]['weight']
            weight_sum += keyword_weight
            keyword_ratio = weight_sum / weight_total
            if keyword_ratio >= keyword_limit:
                # keyword delete
                self.G.remove_node(keyword[0])
        print("Keyword selection finished")
        pass

    def community_detection(self, weight_limit):
        # using louvain's modularity maximization
        node_modularity = nx.algorithms.community.louvain_communities(self.G, weight="weight", resolution=1, seed=42)
        node_modularity = sorted(node_modularity, key=lambda x: len(x), reverse=True)
        self.modularity_dict = {}
        for i, community in enumerate(node_modularity):
            self.modularity_dict[i] = community

        # community classification
        for index in self.modularity_dict.copy():
            node_list = list(self.modularity_dict[index])
            # sort node list by pagerank
            node_list = sorted(node_list, key=lambda x: self.G.nodes[x]['pagerank'], reverse=True)
            community_name = ' '.join(node_list[:5])
            for node in node_list:
                self.G.nodes[node]['community'] = community_name
            self.modularity_dict[community_name] = node_list
            del self.modularity_dict[index]

        # delete community having under 1% of total weight
        Total_weight = sum([self.G.nodes[key]['weight'] for key, value in self.G.nodes.data()])

        for index in self.modularity_dict.copy():
            community_weight = sum([self.G.nodes[key]['weight'] for key in self.modularity_dict[index]])

            if community_weight / Total_weight * 100 < weight_limit:
                print(f"Community {index} is deleted")
                self.G.remove_nodes_from(self.modularity_dict[index]) 
                del self.modularity_dict[index]
        print("total community number: ", len(self.modularity_dict))

        # save network
        nx.write_gexf(self.G, f"{self.DB_path}/KARS_community.gexf")

        print("Community detection finished")

    def research_maturity(self, min_year):
        def gaussian(x, a, b, c, d):
            """Gaussian function to be fitted"""
            return a * np.exp(-(x - b)**2 / (2 * c**2)) + d

        def interactive_graph(x,y,x_est,y_est,PLC_classification):
            p = figure(title=f"Research maturity in {self.DB_name}", x_axis_label='Year', y_axis_label='Keyword frequencies', height=600, width=1000)
            y_max = np.max(y)

            # make bar graph
            p.vbar(x=x, top=y, width=0.5, color='#98A9D7', alpha=0.5, legend_label='Keyword frequencies', hover_alpha=1)

            # add 2nd line for xest and yest
            p.line(x=x_est, y=y_est, line_width=2, color='#F17E6C', line_dash='dashed', alpha=0.7, legend_label='Gaussian fitting')

            # title size
            p.title.text_font_size = '20pt'
            # x,y label no italian
            p.xaxis.axis_label_text_font_style = 'normal'
            p.yaxis.axis_label_text_font_style = 'normal'
            # x,y label size
            p.xaxis.axis_label_text_font_size = '15pt'
            p.yaxis.axis_label_text_font_size = '15pt'
            # no grid
            p.xgrid.grid_line_color = None
            p.ygrid.grid_line_color = None
            # xrange
            p.x_range.start = np.min(x_est)
            p.x_range.end = np.max(x_est)
            # yrange
            p.y_range.start = 0
            p.y_range.end = y_max * 1.1
            # hide legend
            p.legend.visible = False
            # legend location at out of graph 
            legend = Legend(items=[
                ("Keyword frequencies", [p.renderers[0]]),
                ("Gaussian fitting", [p.renderers[1]])
            ], location="center", orientation="horizontal", click_policy="hide")
            p.add_layout(legend, 'below')
            
   
            # add PLC classification as vertical line
            for PLC in PLC_classification:
                PLC_line = Span(location=PLC_classification[PLC][0], dimension='height', line_color='gray', line_width=0.5)
                p.add_layout(PLC_line)

                PLC_label = Label(x=PLC_classification[PLC][0], y=(y_max * 1.1), text=f'{PLC} ({PLC_classification[PLC][0]})', text_font_size='10pt', text_color='gray', angle=90, angle_units='deg', text_baseline='top', text_align='right')
                p.add_layout(PLC_label)


            # add hover tool
            p.toolbar.active_drag = None  # 이동 동작 금지
            hover = HoverTool()
            hover.tooltips = [("Year", "@x"), ("Freq", "@top")]
            p.add_tools(hover)

            # save as html
            output_file(f"{self.DB_path}/research_maturity.html")
            save(p)

            return p

        print("Research maturity started")
        # extract year weight from self.G
        year_weight = {}
        for node in self.G.nodes:
            for year in self.G.nodes[node].keys():
                try:
                    int(year)
                except:
                    continue

                if not year in year_weight:
                    year_weight[str(year)] = 0
                year_weight[str(year)] += int(self.G.nodes[node][str(year)])

        # sort year_weight by year
        year_weight = sorted(year_weight.items(), key=lambda x: x[0], reverse=False)

        # remove year over now year
        now_year = datetime.datetime.now().year
        year_weight = [year for year in year_weight if int(year[0]) < now_year]
        if min_year:
            year_weight = [year for year in year_weight if int(year[0]) >= min_year]

        # extract x, y from year_weight
        min_year = int(year_weight[0][0])
        x = np.array([int(year[0]) for year in year_weight])
        y = np.array([int(year[1]) for year in year_weight])

        # Estimate the initial guess
        amplitude = np.max(y)  # Estimate the amplitude
        mean = np.sum(x * y) / np.sum(y)  # Estimate the mean
        std_dev = np.sqrt(np.sum(y * (x - mean)**2) / np.sum(y))  # Estimate the standard deviation
        offset = np.mean(y)  # Estimate the offset

        # Fit the Gaussian function to the data using curve_fit with the initial guess
        p0 = [amplitude, mean, std_dev, offset]
        # 10000번 반복하고, optimization 값 관계 없이 추정값을 반환하도록 설정
        popt, pcov = curve_fit(gaussian, x, y, p0=p0, maxfev=10000, method='lm', ftol=1e-2, xtol=1e-2, gtol=1e-2)
        y_fit = gaussian(x, *popt)
        mu = popt[1]
        sigma = popt[2]

        # predict future year until mu+3*sigma
        x_est = np.arange(min_year, int(mu+3*sigma)+1)
        y_est = gaussian(x_est, *popt)

        # classificate the year 
        self.PLC_classification = {
            "development": (min_year,int(mu-3*sigma)),
            "introduction": (int(mu-3*sigma),int(int(mu-1.5*sigma))),
            "growth": (int(mu-1.5*sigma),int(int(mu-0.5*sigma))),
            "maturity": (int(mu-0.5*sigma),int(mu+0.5*sigma)),
            "decline": (int(mu+0.5*sigma),int(mu+3*sigma)),
        }

        # plot the graph
        plot = interactive_graph(x,y,x_est,y_est, self.PLC_classification)

        print("Research maturity finished")
        return plot

    def community_year_trend(self, start_PLC):
        def interactive_graph(community_year_weight, PLC_classification):
            p = figure(title=f"community year trend of {self.DB_name}", x_axis_label='Year', y_axis_label='Keyword frequencies %',height=400, width=1000)
            y_max = 0

            # make line graph which y is % of total
            for i, community in enumerate(community_year_weight):
                x = [ year for year, weight in community_year_weight[community].items()]
                y = [ weight for year, weight in community_year_weight[community].items()]
                x_min = min(x)
                x_max = max(x)
                p.line(x=x, y=y, line_width=2, legend_label=f'{community}', line_color=Category20[20][i], name=f'{community}')
                y_max = max(y_max, max(y))

            # figure size
            # title size
            p.title.text_font_size = '20pt'
            # x,y label no italian
            p.xaxis.axis_label_text_font_style = 'normal'
            p.yaxis.axis_label_text_font_style = 'normal'
            # x,y label size
            p.xaxis.axis_label_text_font_size = '15pt'
            p.yaxis.axis_label_text_font_size = '15pt'
            # no grid
            p.xgrid.grid_line_color = None
            p.ygrid.grid_line_color = None
            # xrange
            p.x_range.start = int(x_min)
            p.x_range.end = int(x_max)
            # yrange
            p.y_range.start = 0
            p.y_range.end = y_max * 1.1
            # hide legend
            p.legend.visible = False
            # legend location at out of graph 
            legend = Legend(items=[(community, [p.renderers[i]]) for i, community in enumerate(community_year_weight)], location=(0, 0))
            p.add_layout(legend, 'right')
            
   
            # add PLC classification as vertical line
            for PLC in PLC_classification:
                PLC_line = Span(location=PLC_classification[PLC][0], dimension='height', line_color='gray', line_width=0.5)
                p.add_layout(PLC_line)

                PLC_label = Label(x=PLC_classification[PLC][0], y=(y_max * 1.1), text=f'{PLC} ({PLC_classification[PLC][0]})', text_font_size='10pt', text_color='gray', angle=90, angle_units='deg', text_baseline='top', text_align='right')
                p.add_layout(PLC_label)


            # add hover tool
            p.toolbar.active_drag = None  # 이동 동작 금지
            hover = HoverTool()
            hover.tooltips = [("Year", "@x"), ("Freq %", "@y"), ("Community", "$name")]
            p.add_tools(hover)

            # save as html
            output_file(f"{self.DB_path}/community_year_trend.html")
            save(p)

            return p

        print("Community trend started")
        # extract year weight from self.G
        community_year_weight = {}
        for node in self.G.nodes:
            community = self.G.nodes[node]['community']
            if not community in community_year_weight:
                community_year_weight[community] = {}
            for year in self.G.nodes[node].keys():
                try:
                    int(year)
                except:
                    continue

                if not year in community_year_weight[community].keys():
                    community_year_weight[community][str(year)] = 0
                community_year_weight[community][str(year)] += int(self.G.nodes[node][str(year)])

        # Get all year list
        year_list = []
        for community in community_year_weight:
            year_list += list(community_year_weight[community].keys())
        year_list = list(set(year_list))
        year_list.sort()

        # fill empty year with 0
        for community in community_year_weight:
            for year in year_list:
                if not year in community_year_weight[community].keys():
                    community_year_weight[community][year] = 0

        # remove year over now year and under growth year
        now_year = datetime.datetime.now().year
        start_PLC_year = self.PLC_classification[start_PLC][0]
        for community in community_year_weight:
            for year in community_year_weight[community].copy():
                if int(year) >= now_year or int(year) < start_PLC_year:
                    del community_year_weight[community][year]

        # sort community by sum weight
        community_year_weight = dict(sorted(community_year_weight.items(), key=lambda item: sum(item[1].values()), reverse=True))

        # sort year
        for community in community_year_weight:
            community_year_weight[community] = dict(sorted(community_year_weight[community].items(), key=lambda item: item[0]))

        # Get total year weight
        total_year_weight = {}
        for community in community_year_weight:
            for year in community_year_weight[community]:
                if not year in total_year_weight:
                    total_year_weight[year] = 0
                total_year_weight[year] += community_year_weight[community][year]

        # make % of total
        for community in community_year_weight.copy():
            for year in community_year_weight[community]:
                community_year_weight[community][year] = round(community_year_weight[community][year] / total_year_weight[year] * 100, 2)

        # plot the graph
        plot = interactive_graph(community_year_weight, self.PLC_classification)
        print("Community trend finished")

        return plot

    def keyword_evolution(self, top_rank, start_PLC, end_PLC):
        """
        community 별 Keyword 변화량 List up
        """

        def interactive_graph(keyword_evolution):
            # find x max for all change
            x_max = 0
            for community in keyword_evolution.keys():
                for type in keyword_evolution[community].keys():
                    for node in keyword_evolution[community][type]:
                        change = keyword_evolution[community][type][node]
                        if abs(change) > x_max:
                            x_max = abs(change)

            # make line graph which y is % of total
            plot_list = []
            for i, community in enumerate(keyword_evolution.keys()):
                increase_data = {"keyword": [], "change": [], "rank": []}
                decrease_data = {"keyword": [], "change": [], "rank": []}

                # 더 긴 쪽을 기준으로 y_range 설정
                if len(keyword_evolution[community]["increase"]) <= len(keyword_evolution[community]["decrease"]):
                    y_range = decrease_data["rank"]
                else:
                    y_range = increase_data["rank"]

                # 데이터셋 생성
                rank = 1
                for keyword in keyword_evolution[community]["increase"]:
                    increase_data["keyword"].append(keyword)
                    increase_data["change"].append(keyword_evolution[community]["increase"][keyword])
                    increase_data["rank"].append(str(rank))
                    rank += 1
                rank = 1
                for keyword in keyword_evolution[community]["decrease"]:
                    decrease_data["keyword"].append(keyword)
                    decrease_data["change"].append(keyword_evolution[community]["decrease"][keyword])
                    decrease_data["rank"].append(str(rank))
                    rank += 1

                # ColumnDataSource 생성
                source_increase = ColumnDataSource(data=increase_data)
                source_decrease = ColumnDataSource(data=decrease_data)
                p = figure(title=community, y_range=y_range, x_axis_label='frequency change %', height=600, width=600)

                # 가로 막대 그래프 그리기 (왼쪽 데이터셋)
                bars_increase = p.hbar(y='rank', right='change', height=0.4, color='green', source=source_increase)

                # 가로 막대 그래프 그리기 (오른쪽 데이터셋)
                bars_decrease = p.hbar(y='rank', right='change', height=0.4, color='red', source=source_decrease)

                # figure size
                # title size
                p.title.text_font_size = '10pt'
                # x,y label no italian
                p.xaxis.axis_label_text_font_style = 'normal'
                # no y axis
                p.yaxis.visible = False
                # x,y label size
                p.xaxis.axis_label_text_font_size = '10pt'
                # xrange
                p.x_range.start = -x_max * 1.5
                p.x_range.end = x_max * 1.5
                # no grid
                p.xgrid.grid_line_color = None
                p.ygrid.grid_line_color = None
                # add text
                p.text(x='change', y='rank', text='keyword', source=source_increase, text_align='left', text_baseline='middle', text_font_size="10pt")
                p.text(x='change', y='rank', text='keyword', source=source_decrease, text_align='right', text_baseline='middle', text_font_size="10pt")
                # add vertical line at x= 0
                p.line([0, 0], [0, len(y_range)], line_color="black", line_width=1)

                # add hover tool
                p.toolbar.active_drag = None  # 이동 동작 금지
                hover = HoverTool()
                hover.tooltips = [("keyword", "@keyword"), ("Freq %", "@change")]
                p.add_tools(hover)

                # 탭 생성
                tab = TabPanel(child=p, title=f"community {i}")
                plot_list.append(tab)
        
            # 탭 생성
            plot = Tabs(tabs=plot_list)

            # save as html
            output_file(f"{self.DB_path}/keyword_evolution.html")
            save(plot)
            
            return plot

        print("keyword_evolution started")
        # find top keywords in each community
        now_year = datetime.datetime.now().year
        start_PLC_year = self.PLC_classification[start_PLC]
        start_total = 0
        end_PLC_year = self.PLC_classification[end_PLC]
        end_total = 0
        top_keywords = {}
        for community in self.modularity_dict.keys():
            top_keywords[community] = {}
            node_list = self.modularity_dict[community]
            for node in node_list[:top_rank]:
                top_keywords[community][node] = {}
                for year in self.G.nodes[node].keys():
                    try:
                        int(year)
                    except:
                        continue

                    if int(year) >= now_year:
                        continue

                    if int(year) in range(start_PLC_year[0], start_PLC_year[1]):
                        if not "intro_weight" in top_keywords[community][node]:
                            top_keywords[community][node]["intro_weight"] = 0
                        top_keywords[community][node]["intro_weight"] += self.G.nodes[node][year]
                        start_total += self.G.nodes[node][year]
                        continue
                    
                    elif int(year) in range(end_PLC_year[0], end_PLC_year[1]):
                        if not "maturity_weight" in top_keywords[community][node]:
                            top_keywords[community][node]["maturity_weight"] = 0
                        top_keywords[community][node]["maturity_weight"] += self.G.nodes[node][year]
                        end_total += self.G.nodes[node][year]
                        continue

        # Get keyword evolution
        keyword_evolution = {}
        for community in top_keywords.keys():
            if not community in keyword_evolution:
                keyword_evolution[community] = {"increase": {}, "decrease": {}}
            for node, weight in top_keywords[community].items():
                if "intro_weight" in weight:
                    intro_percent = weight["intro_weight"] / start_total * 100
                else:
                    intro_percent = 0
                if "maturity_weight" in weight:
                    maturity_percent = weight["maturity_weight"] / end_total * 100
                else:
                    maturity_percent = 0
                if intro_percent == 0 and maturity_percent == 0:
                    continue
                change_percent = round(maturity_percent - intro_percent, 2)

                if change_percent > 0:
                    keyword_evolution[community]["increase"][node] = change_percent
                elif change_percent < 0:
                    keyword_evolution[community]["decrease"][node] = change_percent

        # sort keyword by change
        for community in keyword_evolution:
            keyword_evolution[community]["increase"] = dict(sorted(keyword_evolution[community]["increase"].items(), key=lambda item: item[1], reverse=True))
            keyword_evolution[community]["decrease"] = dict(sorted(keyword_evolution[community]["decrease"].items(), key=lambda item: item[1], reverse=False))

        # plot the graph
        plot = interactive_graph(keyword_evolution)
        print("keyword_evolution finished")

        return plot