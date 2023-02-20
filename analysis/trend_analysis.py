import os
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
from scipy.stats import lognorm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

class trend_analysis:
    def __init__(self,save_path, fit_type):
        self.save_path = save_path
        self.fit_type = fit_type
        self.keywords_freq_dict = {}
        with open(f"{self.save_path}/node_feature.json", 'r') as f:
            self.node_feature = json.load(f)
        self.keywords_freq_year()
        self.trend_interpolation()
        self.plot_total_year_trend()
        self.plot_community_year_trend()
        self.save_keywords_freq_dict()
    
    def __call__(self,save_path, fit_type):
        self.save_path = save_path
        self.fit_type
        self.keywords_freq_dict = {}
        self.keywords_freq_year()
        self.trend_interpolation()
        self.plot_total_year_trend()
        self.plot_community_year_trend()
        self.save_keywords_freq_dict()
        
    def keywords_freq_year(self):
        # 1. Get community keywords
        for community in os.listdir(f"{self.save_path}/"):
            if os.path.isdir(f"{self.save_path}/{community}"):
                with open(f"{self.save_path}/{community}/graph.json", 'r') as f:
                    subgraph = json.load(f)
                if not community in self.keywords_freq_dict.keys():
                    self.keywords_freq_dict[community] = {"keywords":[], "year_freq":{}, "year_percent":{}}
                self.keywords_freq_dict[community]["keywords"] = [node["id"] for node in subgraph["nodes"]]

        # 2. Get keywords frequency per year
        for keyword, value in tqdm(self.node_feature.items()):
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

    def gaussian_fit(self, year_list, freq_list, lr=0.1, epochs=10000, verbose=False):
        # define the model and input data
        class SingleGaussianModel(torch.nn.Module):
            def __init__(self):
                super(SingleGaussianModel, self).__init__()
                self.mu = torch.nn.Parameter(torch.tensor([0.]))
                self.sigma = torch.nn.Parameter(torch.tensor([1.]))
                self.coeff = torch.nn.Parameter(torch.tensor([1.]))
                
            def forward(self, x):
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x, dtype=torch.float)
                return self.coeff * torch.exp(- (x - self.mu) ** 2 / (2 * self.sigma ** 2))
            
            def coeff_init(self, x, y):
                self.coeff = torch.nn.Parameter(torch.tensor([y.max()]))
                self.mu = torch.nn.Parameter(torch.tensor([x[y.argmax()]]))
                self.sigma = torch.nn.Parameter(torch.tensor([1.]))
                
        # Loss function
        def loss_func(x, pred):
            return F.mse_loss(x, pred)

        x = torch.tensor(year_list, dtype=torch.float)
        y = torch.tensor(freq_list, dtype=torch.float)

        model = SingleGaussianModel()
        model.coeff_init(x, y)
        optim = torch.optim.Adam(model.parameters(), lr=lr)

        # Training loop
        epochs = epochs
        for i in range(epochs):
            pred = model(x) # forward pass
            loss = loss_func(y, pred) # calculate the loss
            optim.zero_grad() # zero grads
            loss.backward() # backward pass
            optim.step() # update weights
            if verbose:
                if i%100 == 0:
                    print('Epoch: {}, Loss: {}'.format(i, loss.item()), model.mu.item(), model.sigma.item(), model.coeff.item())
        return {'mu': model.mu.item(), 'sigma': model.sigma.item(), 'coeff': model.coeff.item(), 'model': model}

    def trend_interpolation(self):
        # 1. get total keywords frequency per year
        # make x as range from 0 to len(year_freq)
        x = np.linspace(0, len(self.keywords_freq_dict["total"]["year_freq"]), len(self.keywords_freq_dict["total"]["year_freq"]), endpoint=False)
        y = np.array(list(self.keywords_freq_dict["total"]["year_freq"].values())).astype(float)

        # 2. curve fit
        if self.fit_type == "gaussian":
            def gaussian(x, a, b, c, d):
                """Gaussian function to be fitted"""
                return a * np.exp(-(x - b)**2 / (2 * c**2)) + d

            # Estimate the initial guess
            amplitude = np.max(y)  # Estimate the amplitude
            mean = np.sum(x * y) / np.sum(y)  # Estimate the mean
            std_dev = np.sqrt(np.sum(y * (x - mean)**2) / np.sum(y))  # Estimate the standard deviation
            offset = np.mean(y)  # Estimate the offset

            # Fit the Gaussian function to the data using curve_fit with the initial guess
            p0 = [amplitude, mean, std_dev, offset]
            popt, pcov = curve_fit(gaussian, x, y, p0=p0)
            y_fit = gaussian(x, *popt)

            mu = popt[1]
            sigma = popt[2]
            
        elif self.fit_type == "lognorm":
            # transform 0 to 0.00001
            y_ln = y
            y_ln[y==0] = 0.0000001

            # transform y to log(y)
            y_ln = np.log(y_ln)

            def gaussian(x, a, b, c):
                return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))
            popt, pcov = curve_fit(gaussian, x, y_ln, p0=[1, 1, 1])

            # transform y back to exp(y)
            y_fit = np.exp(gaussian(x, *popt))

            mu = popt[1]
            sigma = popt[2]

        elif self.fit_type == "gu":
            # import necessary packages
            output = self.gaussian_fit(x, y, lr=5, epochs=10000, verbose=True)
            mu, sigma, coeff = output['mu'], output['sigma'], output['coeff']
            model = output['model']
            y_fit = model(x).detach().numpy()

        # 3. get mu-3sigma, mu-sigma, mu+sigma, mu+3sigma
        min_year = self.keywords_freq_dict["total"]["min_year"]
        mu_3sigma = mu - 3*sigma + min_year
        mu_2sigma = mu - 2*sigma + min_year
        mu = mu + min_year
        mu_plus_sigma = mu + sigma + min_year
        PLC = {"development":mu_3sigma, "introduction":mu_2sigma, "growth":mu, "maturity":mu_plus_sigma}

        # 4. save PLC to self.keywords_freq_dict
        self.keywords_freq_dict["total"]["PLC"] = PLC

        # 5. plot gaussian interpolation with total keywords frequency per year but x axis is year
        plt.subplots(figsize=(10, 5))
        x = [int(year) for year in list(self.keywords_freq_dict["total"]["year_freq"].keys())]
        plt.plot(x, y, 'b+:', label='data')
        plt.xlim(x[0], x[-1])
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
        plt.savefig(f"{self.save_path}/gaussian_interpolation.png")

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
        legend = list(self.keywords_freq_dict.keys())
        legend.remove("total")
        legend.append("PLC")
        plt.legend(legend, loc='upper left')
        plt.grid(alpha=0.5, linestyle='--')
        plt.xlabel("Year")
        plt.ylabel("Keywords Frequency")
        plt.minorticks_on()
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/total_year_trend.png")

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
        plt.title("comunities trend per Year")
        plt.xlim(int(self.keywords_freq_dict["total"]["PLC"]["development"]), int(self.keywords_freq_dict["total"]["max_year"]))
        plt.gca().set_ylim(bottom=0)
        plt.xticks(np.arange(int(self.keywords_freq_dict["total"]["PLC"]["development"]), int(self.keywords_freq_dict["total"]["max_year"]), 5))
        legend = list(self.keywords_freq_dict.keys())
        legend.remove("total")
        legend.append("PLC")
        plt.legend(legend, loc='upper left')
        plt.grid(alpha=0.5, linestyle='--')
        plt.xlabel("Year")
        plt.ylabel("% of Keywords Frequency")
        plt.minorticks_on()
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/community_year_trend.png")

    def save_keywords_freq_dict(self):
        with open(f"{self.save_path}/keywords_freq_dict.json", "w") as f:
            json.dump(self.keywords_freq_dict, f, indent=4)