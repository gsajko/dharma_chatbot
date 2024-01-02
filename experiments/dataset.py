# %%

import pandas as pd

# %%
# read in csv file
csv_file = "data/Rob_Burbea_Transcripts.2023-12-31.csv"
df = pd.read_csv(csv_file)

df.columns = df.columns.str.replace(" ", "_").str.lower()


# %%
# split the transcript_or_writing column into pdf name and create new column
# remove .pdf from pdf_name
df["name"] = df.transcript_or_writing.str.split("/").str[-1].str.replace(".pdf", "")

# %%
# drop first row
df = df.drop(df.index[0])
# %%

# %%
cols = [
    "name",
    "date",
    "title_of_event",
    "title_of_talk_or_writing",
    "broad_topics",
    "detailed_topics",
    "length_of_recording",
    "type_of_recording",
]
df[cols].head()
# %%
df.type_of_recording.unique()
# %%
