from KARS import KARS

if __name__ == "__main__":
    # name the database
    DB_name = "HE_steel"
    #DB_name = "ReRAM"

    # load KARs
    kars = KARS(DB_name)

    # collect metadata
    engine_list = ["crossref", "openalex"]
    # keyword_list = [
    #     "RRAM",
    #     "ReRAM",
    #     "OxRAM",
    #     "OxRRAM",
    #     "CBRAM",
    #     "Resistive memory",
    #     "resistance memory",
    #     "Valence change memory",
    #     "redox-based memory",
    #     "Resistive switch",
    #     "resistance switch",
    #     "filament switch",
    #     "filamentary switch",
    #     "complementary switch",
    #     "Bipolar switch",
    #     "Unipolar switch",
    #     "conductive filament",
    #     "oxygen vacancies filament",
    #     "oxygen vacancy filament",
    #     "electroforming filament",
    #     "Electrochemical metallization",
    #     "conductive bridge",
    #     "quantized conductance",
    # ]

    keyword_list = [
        "hydrogen embrittlement steel"
    ]

    # collect metadata
    #kars.collect_metadata(engine_list, keyword_list)

    # construct PSPP network
    #kars.construct_PSPP_network()

    # research trend analysis
    kars.research_trend_analysis()

    # PSPP relation analysis
    #kars.PSPP_relation_analysis()
    


    