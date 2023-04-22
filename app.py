from spacy import displacy
import yaml
import gradio as gr
from extraction import ner_extract
import pandas as pd

class web_visualizer:
    def __init__(self):
        self.ner_extracter = ner_extract()
        config = 'configs/config.yaml'
        self.config_dict = yaml.load(open(config, 'r'), Loader=yaml.FullLoader)

    def NER_visual(self,tokens,labels,top_probs):
        NER_dict = {}
        NER_text = ""
        NER_ents = []

        for token, label in zip(tokens, labels):
            if label == "O":
                NER_text += " " + token
            else:
                start = len(NER_text)
                NER_text += " " + token
                end = len(NER_text)
                NER_ents.append({"start":start, "end":end, "label": label})

        NER_dict = {"text": NER_text, "ents": NER_ents, "title": None}

        return NER_dict
    
    def get_labels(self,text):
        output = self.ner_extracter.extract(text)[0]
        NER_dict = self.NER_visual(output['token'],output['predicted_tag'],output['top_prob'])
        option = self.config_dict["visualization"]
        html = displacy.render(NER_dict, style="ent", manual=True, options=option, page=True)
        return html
    
    def get_article(self):
        description_df = './dataset/NER_labels.csv'
        df = pd.read_csv(description_df)
        df_html = df.to_html(index=False)
        description = f"<h3>The meaning of the labels is as follows:</h3>{df_html}"
        return description

    def get_description(self):
        text = "<b>본 모델은 ReRAM 문헌에서 나타나는 단어들을 19개의 class로 분류하기 위해 개발된 모델입니다.<br> 잘못된 결과가 출력될 경우 플래그 버튼을 통해 error reporting을 해주시면 감사하겠습니다 :)</b>"
        description = f'<p style="font-size:16px">{text}</p>'
        return description

    def visualize(self):
        demo = gr.Interface(
            fn=self.get_labels, 
            inputs=gr.Textbox(placeholder="Enter sentence here..."), 
            outputs="html", 
            title="NER Demo", 
            allow_flagging=True,
            examples=[
                      "In this letter, we observed a different type of current overshoot during the RESET process. The RESET current overshoot was confirmed to have severe effects on the endurance of RRAM. We also demonstrated the relation between the current overshoot and the intrinsic capacitive elements of each state of RRAM. Finally, an optimized pulse shape was proposed to minimize the current overshoot and was experimentally verified to significantly improve the variability and endurance in a typical RRAM device with a W/Zr/HfO2/TiN structure.",
                      "Preliminary characterization of forming, SET, and RESET and a short cycling were performed in quasi-static (QS) condition using an Agilent B1500 semiconductor parameter analyzer (sweep speed of 1 V/s). Forming and switching are shown in Fig.2 for samples S1, S2, and S3, respectively. The average values of the forming voltage (VF), RESET voltage (VRESET), and SET voltage (VSET) were 4, 0.7, and 8 V, respectively, for the S2 and S3 samples (bipolar behavior)."
                      ],
            article=self.get_article(),
            description=self.get_description(),
            )
        demo.launch(share=True)

visualizer = web_visualizer()
visualizer.visualize()