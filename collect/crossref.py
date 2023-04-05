#https://github.com/fabiobatalha/crossrefapi
import os
from crossref.restful import Works, Etiquette
from pylatexenc.latex2text import LatexNodes2Text
import json
from tqdm import tqdm
import subprocess
import pandas as pd

class crossref:
    def __init__(self, keywords, save_path, DB_name, email):
        self.edit_api_setting()
        self.mail = email
        self.my_etiquette = Etiquette('Memory_Trend', '0.0.1', 'My Project URL', self.mail)
        self.API = Works(etiquette=self.my_etiquette)
        self.keyword_list = [ {"search_words": word.split(" "), "stop_words": []} for word in keywords ]
        for keywords in self.keyword_list:
            if "" in keywords["search_words"]:
                keywords["search_words"].remove("") # remove empty string
            for word in keywords["search_words"]:
                if word.startswith("-"):
                    keywords["stop_words"].append(word[1:])
                    keywords["search_words"].remove(word)
        self.save_path = save_path
        self.DB_name = DB_name
        if os.path.exists(f"{self.save_path}/{self.DB_name}/{self.DB_name}.jsonl"):
            os.remove(f"{self.save_path}/{self.DB_name}/{self.DB_name}.jsonl")
        self.Article_search_keyword()
        self.save_metadata()

    def edit_api_setting(self):
        """
        Modify the Crossref API Setting to increase the number of requests
        :return:
        """
        LIMIT = 1000
        MAXOFFSET = 1000000
        FACETS_MAX_LIMIT = 1000000

        envs = subprocess.run(["conda info --system | grep 'CONDA_PREFIX' | head -n 1 | cut -d':' -f2"],shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8').strip() + '/lib/'
        python_path = subprocess.run([f"ls {envs} | grep python | tail -n 1"],shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8').strip() + '/site-packages'
        envs_path = envs+python_path
        
        subprocess.run([f"sed -i -e 's/LIMIT = .*/LIMIT = {LIMIT}/' -e 's/MAXOFFSET = .*/MAXOFFSET = {MAXOFFSET}/' -e 's/FACETS_MAX_LIMIT = .*/FACETS_MAX_LIMIT = {FACETS_MAX_LIMIT}/' {envs_path}/crossref/restful.py"], shell=True)

    def Article_search_keyword(self):
        self.total_query = []
        for keywords in self.keyword_list:
            #1. search metadata with keyword
            print(f"\nNow Searching for {keywords['search_words']} exclude for {keywords['stop_words']} \n")
            keyword = keywords["search_words"]
            stopword = keywords["stop_words"]

            #2. find the minimum number of search results
            count = self.API.query(bibliographic=keyword[0]).filter(type="journal-article").count()
            for word in keyword:
                if count >= self.API.query(bibliographic=word).filter(type="journal-article").count():
                    min_word = word

            #3. search metadata with minimum number of search results
            stop_list = []
            multi_list = []
            with tqdm(total= self.API.query(bibliographic=min_word).filter(type="journal-article").count()) as pbar:
                query = self.API.query(bibliographic=min_word).filter(type="journal-article").select("DOI","title","container-title","publisher","published-print","published-online", "abstract")
                for q in query:
                    # 3. check if title is empty
                    if "title" not in q.keys():
                        pbar.update(1)
                        continue
                    
                    title = LatexNodes2Text().latex_to_text(str(q['title'][0])).replace("\n"," ").lower()
                    # 4. check if the search results contain stopword
                    stop_check = "X"
                    for word in stopword:
                        if title.find(word.lower()) == -1:
                            stop_check = "O"
                            break
                    if stop_check == "O":
                        stop_list.append(q['title'][0])
                        pbar.update(1)
                        continue
                        

                    # 5. check if the search results contain the keyword
                    multi_check = 0
                    for word in keyword:
                        if title.find(word.lower()) != -1:
                            multi_check+=1
                    if multi_check == len(keyword):
                        self.total_query.append(q)
                        pbar.update(1)
                    else:
                        multi_list.append(q['title'][0])
                        pbar.update(1)

            print(f"Deleted {len(stop_list)} articles because of stopword\n")
            print(f"stop_list examples: {stop_list[:5]}\n")
            print(f"Deleted {len(multi_list)} articles because of not enough keywords\n")
            print(f"multi_list examples: {multi_list[:5]}\n")

            #6. delete duplicates within search results
            count=0
            for i, result in enumerate(self.total_query):
                for j, result2 in enumerate(self.total_query):
                    if result["DOI"] == result2["DOI"] and i != j:
                        self.total_query.pop(j)
                        count+=1
            print(f"Deleted {count} duplicates\n")
        
    def save_metadata(self):
        #1. save keyword list
        with open(f"{self.save_path}/{self.DB_name}/search_keyword_list.txt",encoding="utf-8",mode="w") as f:
            for i in self.keyword_list: f.write(f"'{i}'" + "\n")

        #2. save search results
        with open(f"{self.save_path}/{self.DB_name}/{self.DB_name}.jsonl",encoding="utf-8",mode="a") as f:
            for i in self.total_query: f.write(json.dumps(i) + "\n")
        print(f"Saved {len(self.total_query)} articles\n")

        #3. save search results as csv

        df = pd.DataFrame(columns=["Index","Title","Abstract","Published date","DOI"])
        for metadata in self.total_query:
            if "abstract" not in metadata.keys():
                metadata["abstract"] = ""
            if "published-print" in metadata.keys():
                published_date = metadata["published-print"]["date-parts"][0]
            elif "published-online" in metadata.keys():
                published_date = metadata["published-online"]["date-parts"][0]
            else:
                published_date = ""
            df = df.append({"Index":len(df),"Title":metadata["title"][0],"Abstract":metadata["abstract"],"Published date":published_date,"DOI":metadata["DOI"]},ignore_index=True)
        df.to_csv(f"{self.save_path}/{self.DB_name}/{self.DB_name}.csv", index=False)

