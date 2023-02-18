import pandas as pd
from tqdm import tqdm
import re

def Concat_df():
    for i in range(12):
        df_temp = pd.read_excel(f"savedrecs_{i}.xls",engine='xlrd')
        if i == 0:
            df_total = df_temp
        else:
            df_total = pd.concat([df_total,df_temp])
    df_total.reset_index(drop=True,inplace=True)
    return df_total

def Convert_df(df_total):
    ReRAM = pd.DataFrame(columns=['title','abstract','author','publisher','journal',"affiliation","funder",'date','is-referenced-by-count','reference','subject','type'])
    Month_to_Num = {
        "JAN":1,
        "FEB":2,
        "MAR":3,
        "APR":4,
        "MAY":5,
        "JUN":6,
        "JUL":7,
        "AUG":8,
        "SEP":9,
        "OCT":10,
        "NOV":11,
        "DEC":12
    }

    for index, row in tqdm(df_total.iterrows()):
        if not str(row['Publication Year']) == "nan" :
            Date = str(int(row['Publication Year']))
            if not int(Date) == 2022:
                if not str(row['Publication Date']) == "nan":
                    Month = re.sub('[^a-zA-Z]','',str(row['Publication Date']))
                    for Str in Month_to_Num.keys():
                        if Str in Month:
                            Month = str(Month_to_Num[Str])
                            break
                    Date += f",{Month}"

                ReRAM.loc[index,'title'] = row["Article Title"]
                ReRAM.loc[index,'abstract'] = row['Abstract']
                ReRAM.loc[index,'author'] = row['Author Full Names']
                ReRAM.loc[index,'publisher'] = row['Publisher']
                ReRAM.loc[index,"journal"] = row["Source Title"]
                ReRAM.loc[index,"affiliation"] = row["Affiliations"]
                ReRAM.loc[index,"funder"] = row["Funding Name Preferred"]
                ReRAM.loc[index,'date'] = f"[{Date}]"
                ReRAM.loc[index,'is-referenced-by-count'] = row['Cited Reference Count']
                ReRAM.loc[index,'subject'] = row['WoS Categories']
                ReRAM.loc[index,'type'] = row['Document Type']

    ReRAM.to_csv("ReRAM_DB_Keyword_Date_Document.csv", index=False)

    print(f"\n{ReRAM.shape[0]} results in total")
    print("-----------------------------------------------------")

