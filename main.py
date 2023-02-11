import csv
import pandas as pd

txt_file = r"entrenamiento.txt"
csv_file = r"entrenamiento.csv"

with open(txt_file, "r") as in_text:
    in_reader = csv.reader(in_text, quotechar='"', delimiter='\t',
                     quoting=csv.QUOTE_NONE, skipinitialspace=True)
    with open(csv_file, "w") as out_csv:
        out_writer = csv.writer(out_csv)
        out_writer.writerow(["spam", "message"])
        for row in in_reader:
            out_writer.writerow(row)

df = pd.read_csv("entrenamiento.csv")
df["message"] = df["message"].str.replace('[^a-zA-Z]', ' ')
df["message"] = df["message"].replace(r'\s+', ' ', regex=True)
df["message"] = df["message"].str.lower()
df["spam"] = df["spam"].map({"ham":1, "spam":0})
print(df)

