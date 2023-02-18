import os
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class trend_analysis:
    def __init__(self,save_path,DB_name):
        self.save_path = save_path
        self.DB_name = DB_name
        self.keywords_freq_dict = {}
        self.keywords_freq_year()
        self.gaussian_interpolation()
        self.plot_total_year_trend()
        self.plot_community_year_trend()
        self.save_keywords_freq_dict()
        
    def keywords_freq_year(self):
        # 1. Get community keywords
        for community in os.listdir(f"{self.save_path}/{self.DB_name}/"):
            if os.path.isdir(f"{self.save_path}/{self.DB_name}/{community}"):
                with open(f"{self.save_path}/{self.DB_name}/{community}/subgraph.json", 'r') as f:
                    subgraph = json.load(f)
                if not community in self.keywords_freq_dict.keys():
                    self.keywords_freq_dict[community] = {"keywords":[], "year_freq":{}, "year_percent":{}}
                self.keywords_freq_dict[community]["keywords"] = [node["id"] for node in subgraph["nodes"]]

        # 2. Get keywords frequency per year
        with open(f"{self.save_path}/{self.DB_name}/node_feature.json", 'r') as f:
            node_feature = json.load(f)
        for keyword, value in tqdm(node_feature.items()):
            for community in self.keywords_freq_dict.keys():
                if keyword in self.keywords_freq_dict[community]["keywords"]:
                    for year, freq in value.items():
                        if not year == "total":
                            if not year in self.keywords_freq_dict[community]["year_freq"].keys():
                                self.keywords_freq_dict[community]["year_freq"][year] = 0
                            self.keywords_freq_dict[community]["year_freq"][year] += freq

        # 3. find min & max year from all communities
        min_year = 9999
        max_year = 0
        for community in self.keywords_freq_dict.keys():
            for year in self.keywords_freq_dict[community]["year_freq"].keys():
                if int(year) < min_year:
                    min_year = int(year)
                if int(year) > max_year:
                    max_year = int(year)

        # 4. add 0 to all year
        for community in self.keywords_freq_dict.keys():
            for year in range(min_year, max_year+1):
                if not str(year) in self.keywords_freq_dict[community]["year_freq"].keys():
                    self.keywords_freq_dict[community]["year_freq"][str(year)] = 0

        # 5. sort by year
        for community in self.keywords_freq_dict.keys():
            self.keywords_freq_dict[community]["year_freq"] = dict(sorted(self.keywords_freq_dict[community]["year_freq"].items(), key=lambda x: x[0]))

        # 6. delete data of last year
        for community in self.keywords_freq_dict.keys():
            del self.keywords_freq_dict[community]["year_freq"][str(max_year)]

        # 6. make total keywords frequency per year
        self.keywords_freq_dict["total"] = {"keywords":[], "year_freq":{}, "year_percent":{}, "PLC":{}}
        for year in range(min_year, max_year):
            self.keywords_freq_dict["total"]["year_freq"][str(year)] = 0
        for community in self.keywords_freq_dict.keys():
            if not community == "total":
                for year, freq in self.keywords_freq_dict[community]["year_freq"].items():
                    self.keywords_freq_dict["total"]["year_freq"][year] += freq
        self.keywords_freq_dict["total"]["min_year"] = min_year
        self.keywords_freq_dict["total"]["max_year"] = max_year-1

        # 7. year percent of total 
        for community in self.keywords_freq_dict.keys():
            if not community == "total":
                for year, freq in self.keywords_freq_dict[community]["year_freq"].items():
                    if not self.keywords_freq_dict["total"]["year_freq"][year] == 0:
                        self.keywords_freq_dict[community]["year_percent"][year] = freq/self.keywords_freq_dict["total"]["year_freq"][year]
                    else:
                        self.keywords_freq_dict[community]["year_percent"][year] = 0

    def gaussian_interpolation(self):
        # 1. get total keywords frequency per year
        total_year_freq = self.keywords_freq_dict["total"]["year_freq"]
        total_year_freq = dict(sorted(total_year_freq.items(), key=lambda x: x[0]))
        total_year_freq = np.array(list(total_year_freq.values()))

        # 2. get gaussian interpolation
        x = np.linspace(0, len(total_year_freq), len(total_year_freq), endpoint=False)
        y = total_year_freq
        def gaussian(x, a, b, c):
            return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))
        popt, pcov = curve_fit(gaussian, x, y, p0=[1, 1, 1])
        y_fit = gaussian(x, *popt)

        # 3. get mu-3sigma, mu-sigma, mu+sigma, mu+3sigma
        min_year = self.keywords_freq_dict["total"]["min_year"]
        mu_3sigma = popt[1] - 3*popt[2] + min_year
        mu_2sigma = popt[1] - 2*popt[2] + min_year
        mu = popt[1] + min_year
        mu_plus_sigma = popt[1] + popt[2] + min_year
        PLC = {"development":mu_3sigma, "introduction":mu_2sigma, "growth":mu, "maturity":mu_plus_sigma}

        # 4. save PLC to self.keywords_freq_dict
        self.keywords_freq_dict["total"]["PLC"] = PLC

        # 5. plot gaussian interpolation with total keywords frequency per year but x axis is year
        plt.subplots(figsize=(10, 5))
        x = [int(year) for year in list(self.keywords_freq_dict["total"]["year_freq"].keys())]
        plt.plot(x, total_year_freq, 'b+:', label='data')
        plt.plot(x, y_fit, 'r-', label='fit')
        plt.legend()
        plt.xlabel("Year")
        plt.ylabel("Frequency")
        plt.title("Gaussian Interpolation")

        # 6. make vertical line by mu-3sigma, mu-sigma, mu+sigma, mu+3sigma. label these as development, introduction, growth, and maturity on the top of lines
        plt.axvline(x=mu_3sigma, color='gray', linestyle='--', label="Development")
        plt.axvline(x=mu_2sigma, color='gray', linestyle='--', label="Introduction")
        plt.axvline(x=mu, color='gray', linestyle='--', label="Growth")
        plt.axvline(x=mu_plus_sigma, color='gray', linestyle='--', label="Maturity")
        plt.legend()
        plt.show()
        plt.savefig(f"{self.save_path}/{self.DB_name}/gaussian_interpolation.png")

    def plot_total_year_trend(self):
        # 1. plot keywords frequency per year
        plt.subplots(figsize=(10, 5))
        for i, community in enumerate(self.keywords_freq_dict.keys()):
            if not community == "total":
                if i == 0:
                    xy = self.keywords_freq_dict[community]["year_freq"]
                    xy_old = {}
                    #set all year to 0
                    for year, freq in xy.items():
                        xy_old[year] = 0
                else:
                    xy_old = xy.copy()
                    for year, freq in self.keywords_freq_dict[community]["year_freq"].items():
                        if not year in xy.keys():
                            xy[year] = 0
                            xy_old[year] = 0
                        xy[year] += freq

                # sort by year
                xy = dict(sorted(xy.items(), key=lambda item: item[0]))
                xy_old = dict(sorted(xy_old.items(), key=lambda item: item[0]))
                x = list(xy_old.keys())
                x = [int(i) for i in x]
                yp = list(xy_old.values())
                y = list(xy.values())
                plt.fill_between(x, yp, y, alpha=0.5, cmap=plt.cm.rainbow)

        # 6. make vertical line by mu-3sigma, mu-sigma, mu+sigma, mu+3sigma. label these as development, introduction, growth, and maturity on the top of lines
        plt.axvline(x=self.keywords_freq_dict["total"]["PLC"]["development"], color='gray', linestyle='--', label="Development")
        plt.axvline(x=self.keywords_freq_dict["total"]["PLC"]["introduction"], color='gray', linestyle='--', label="Introduction")
        plt.axvline(x=self.keywords_freq_dict["total"]["PLC"]["growth"], color='gray', linestyle='--', label="Growth")
        plt.axvline(x=self.keywords_freq_dict["total"]["PLC"]["maturity"], color='gray', linestyle='--', label="Maturity")

        # plot graphs
        plt.title("Keywords Frequency per Year")
        plt.xlim(int(self.keywords_freq_dict["total"]["min_year"]), int(self.keywords_freq_dict["total"]["max_year"]))
        plt.gca().set_ylim(bottom=0)
        plt.xticks(np.arange(int(self.keywords_freq_dict["total"]["min_year"]), int(self.keywords_freq_dict["total"]["max_year"]), 5))
        plt.legend(self.keywords_freq_dict.keys(), loc='upper left')
        plt.grid(alpha=0.5, linestyle='--')
        plt.xlabel("Year")
        plt.ylabel("Keywords Frequency")
        plt.minorticks_on()
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/{self.DB_name}/total_year_trend.png")

    def plot_community_year_trend(self):
        # 3. multi plots
        colors = plt.cm.rainbow
        plt.subplots(figsize=(10, 5))
        for i, community in enumerate(self.keywords_freq_dict.keys()):
            if not community == "total":
                # get x into int type
                x = self.keywords_freq_dict[community]["year_percent"].keys()
                x = [int(i) for i in x]
                y = self.keywords_freq_dict[community]["year_percent"].values()
                y = [float(i) for i in y]
                
                # draw lines
                plt.plot(x, y, color=colors(i/len(self.keywords_freq_dict.keys())), label=community)
                
        # 6. make vertical line by mu-3sigma, mu-sigma, mu+sigma, mu+3sigma. label these as development, introduction, growth, and maturity on the top of lines
        plt.axvline(x=self.keywords_freq_dict["total"]["PLC"]["development"], color='gray', linestyle='--', label="Development")
        plt.axvline(x=self.keywords_freq_dict["total"]["PLC"]["introduction"], color='gray', linestyle='--', label="Introduction")
        plt.axvline(x=self.keywords_freq_dict["total"]["PLC"]["growth"], color='gray', linestyle='--', label="Growth")
        plt.axvline(x=self.keywords_freq_dict["total"]["PLC"]["maturity"], color='gray', linestyle='--', label="Maturity")

        # limit x from introduction to maturity
        plt.xlim(int(self.keywords_freq_dict["total"]["PLC"]["development"]), int(self.keywords_freq_dict["total"]["max_year"]))
        plt.gca().set_ylim(bottom=0)
        plt.xticks(np.arange(int(self.keywords_freq_dict["total"]["PLC"]["development"]), int(self.keywords_freq_dict["total"]["max_year"]), 5))
        plt.legend(self.keywords_freq_dict.keys(), loc='upper left')
        plt.grid(alpha=0.5, linestyle='--')
        plt.xlabel("Year")
        plt.ylabel("% of Keywords Frequency")
        plt.minorticks_on()
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/{self.DB_name}/community_year_trend.png")

    def save_keywords_freq_dict(self):
        with open(f"{self.save_path}/{self.DB_name}/keywords_freq_dict.json", "w") as f:
            json.dump(self.keywords_freq_dict, f, indent=4)