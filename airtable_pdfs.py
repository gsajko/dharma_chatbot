# %%
import os

import pandas as pd
import requests

# %%
# read in csv file
csv_file = "data/Rob_Burbea_Transcripts.2023-12-31.csv"
df = pd.read_csv(csv_file)

df.columns = df.columns.str.replace(" ", "_").str.lower()


# %%
def extract_pdf_links(links_list: list):
    pdf_list = []
    for url in links_list:
        try:
            pdf_link = {}
            pdf_link["name"] = url.split("/")[-1]
            pdf_link["url"] = url
            pdf_list.append(pdf_link)
        except Exception as e:
            print(e)
            pass
    return pdf_list


pdf_list = extract_pdf_links(df.transcript_or_writing)


# %%
def download_pdf(pdf_list: list):
    directory = "data/pdf_raw"
    for link in pdf_list:
        # check if pdf exists
        if os.path.exists(f'{directory}/{link["name"]}'):
            print(f'{directory}/{link["name"]} exists')
            pass
        else:
            # download pdf from url, and save in pdf folder
            url = link["url"]
            r = requests.get(url)
            with open(f'{directory}/{link["name"]}', "wb") as f:
                f.write(r.content)
            print(f'{directory}/{link["name"]} created')


download_pdf(pdf_list)
# %%
