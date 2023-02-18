#https://github.com/fabiobatalha/crossrefapi
import os
from crossref.restful import Works, Etiquette
import json
from tqdm import tqdm
import subprocess

class crossref:
    def __init__(self, keywords, save_path, DB_name, email):
        self.mail = email
        self.my_etiquette = Etiquette('Memory_Trend', '0.0.1', 'My Project URL', self.mail)
        self.API = Works(etiquette=self.my_etiquette)
        self.keyword_list = [ word.split(" ") for word in keywords ]
        self.save_path = save_path
        self.DB_name = DB_name
        if os.path.exists(f"{self.save_path}/{self.DB_name}/{self.DB_name}.jsonl"):
            os.remove(f"{self.save_path}/{self.DB_name}/{self.DB_name}.jsonl")
        self.edit_api_setting()

    def edit_api_setting(self):
        """
        Modify the Crossref API Setting to increase the number of requests
        :return:
        """
        LIMIT = 10000
        MAXOFFSET = 1000000
        FACETS_MAX_LIMIT = 1000000

        envs = subprocess.run(["conda info --system | grep 'CONDA_PREFIX' | head -n 1 | cut -d':' -f2"],shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8').strip() + '/lib/'
        python_path = subprocess.run([f"ls {envs} | grep python | tail -n 1"],shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8').strip() + '/site-packages'
        envs_path = envs+python_path
        
        #envs_path = subprocess.run(["conda info --system | grep 'conda location' | cut -d':' -f2 | rev | cut -d'/' -f2- | rev"],shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
        subprocess.run([f"sed -i -e 's/LIMIT = .*/LIMIT = {LIMIT}/' -e 's/MAXOFFSET = .*/MAXOFFSET = {MAXOFFSET}/' -e 's/FACETS_MAX_LIMIT = .*/FACETS_MAX_LIMIT = {FACETS_MAX_LIMIT}/' {envs_path}/crossref/restful.py"], shell=True)

    def Article_search_keyword(self):
        Total_query = []
        for keyword in self.keyword_list:
            #1. search metadata with keyword
            print(f"\nNow Searching for {keyword}\n")
            count = self.API.query(bibliographic=keyword[0]).count()
            for word in keyword:
                if count >= self.API.query(bibliographic=word).count():
                    min_word = word

            with tqdm(total= self.API.query(bibliographic=min_word).count()) as pbar:
                query = self.API.query(bibliographic=min_word)
                for q in query:
                    count = 0
                    try:
                        for word in keyword:
                            if q["title"][0].lower().find(word.lower()) != -1:
                                count+=1
                        if count == len(keyword):
                            Total_query.append(q)
                    except:
                        pass
                    pbar.update(1)

            #2. delete duplicates within search results
            count=0
            for i, result in enumerate(Total_query):
                for j, result2 in enumerate(Total_query):
                    if result["DOI"] == result2["DOI"] and i != j:
                        Total_query.pop(j)
                        count+=1
            print(f"Deleted {count} duplicates\n")
        
            #3. save search results
            with open(f"{self.save_path}/{self.DB_name}/{self.DB_name}.jsonl",encoding="utf-8",mode="a") as f:
                for i in Total_query: f.write(json.dumps(i) + "\n")
            print(f"Saved {len(Total_query)} articles\n")
