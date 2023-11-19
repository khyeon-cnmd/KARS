from KARS import KARS

if __name__ == "__main__":
    # name the database
    DB_path = "/home1/khyeon/Researches/KBSE/ReRAM"

    # load KARs
    kars = KARS(DB_path)

    # extract keywords
    kars.keyword_extraction()

    # construct keyword network
    kars.network_construction()

    # research trend analysis
    kars.research_trend_analysis()

    


    