"""
Lab 3 - Inteligencia Artificial

Fecha de inicio: 10/02/2023
"""

# Imports
import csv
import pandas as pd


def TxtReaderWriter(txt_file, csv_file):
    with open(txt_file, "r") as in_text:
        in_reader = csv.reader(
            in_text,
            quotechar='"',
            delimiter="\t",
            quoting=csv.QUOTE_NONE,
            skipinitialspace=True,
        )
        with open(csv_file, "w") as out_csv:
            out_writer = csv.writer(out_csv)
            out_writer.writerow(["spam", "message"])
            for row in in_reader:
                out_writer.writerow(row)


# Call to function
TxtReaderWriter(r"entrenamiento.txt", r"entrenamiento.csv")

# Read with pandas
df = pd.read_csv("entrenamiento.csv")

# Normalization of data
df["message"] = df["message"].str.replace("[^a-zA-Z]", " ")
df["message"] = df["message"].replace(r"\s+", " ", regex=True)
df["message"] = df["message"].str.lower()

# Map data with hamp and spam
df["spam"] = df["spam"].map({"ham": 1, "spam": 0})


# Ham = 1
# Spam = 0
print(df["spam"].value_counts())
# Print data
