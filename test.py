import os
from zipfile import ZipFile

import gradio as gr


def zip_files(files):
    with ZipFile("tmp.zip", "w") as zipObj:
        for idx, file in enumerate(files):
            zipObj.write(file.name, file.name.split("/")[-1])
    return "tmp.zip"

demo = gr.Interface(
    zip_files,
    gr.File(file_count="multiple", file_types=["text", ".json", ".csv"]),
    "file",
    cache_examples=True
)

if __name__ == "__main__":
    demo.launch()
