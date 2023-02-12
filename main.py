"""
Lab 3 - Inteligencia Artificial

Fecha de inicio: 10/02/2023
"""

# Imports
from collections import defaultdict
import csv
import re
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

# Separate into training and test set
random_data = df.sample(frac=1, random_state=1)
training_test_index = round(len(df) * 0.8)
training_set = random_data[:training_test_index].reset_index(drop=True)
test_set = random_data[training_test_index:].reset_index(drop=True)

# Validation Test
validationTest_index = round(len(test_set) * 0.1)
validation_test = random_data[:validationTest_index].reset_index(drop=True)
validation_set = random_data[validationTest_index:].reset_index(drop=True)


# Saving data into different files

with open("training_set.csv", "w") as training:
    training.write(training_set.to_csv(index=False))
with open("test_set.csv", "w") as test:
    test.write(test_set.to_csv(index=False))
with open("validation_test.csv", "w") as validation:
    validation.write(validation_test.to_csv(index=False))

# Normalization of data
training_set["message"] = training_set["message"].str.replace("[^a-zA-Z]", " ")
training_set["message"] = training_set["message"].replace(r"\s+", " ", regex=True)
training_set["message"] = training_set["message"].str.lower()


# dictionaryProbability function
def dictionaryProbability(data, laplace, data2):
    return {
        palabra: (data[palabra] + laplace) / (data2 + laplace * len(data))
        for palabra in data
    }


# Construction of model
def bayesLaplaceSmoothingModel(training_set):
    training_set["spam"] = training_set["spam"].map({"ham": 1, "spam": 0})

    palabras_spam = 0
    palabras_ham = 0
    paramLaplaceSmoothing = 1
    diccionarioSpam = defaultdict(int)
    diccionarioHam = defaultdict(int)

    for __, r in training_set.iterrows():
        ham = r["spam"] == 1
        spam = r["spam"] == 0
        mensaje = r["message"]
        palabras = re.findall(r"\b\w+\b", mensaje)

        for x in palabras:
            if spam:
                palabras_spam += 1
                diccionarioSpam[x] += 1
            if ham:
                palabras_ham += 1
                diccionarioHam[x] += 1

    # Probabilidades - Laplace Smoothing
    probabilidadSpam = (
        training_set["spam"].value_counts()[0] + paramLaplaceSmoothing
    ) / (len(training_set) + paramLaplaceSmoothing * 2)

    probabilidadHam = (
        training_set["spam"].value_counts()[1] + paramLaplaceSmoothing
    ) / (len(training_set) + paramLaplaceSmoothing * 2)

    psw = dictionaryProbability(diccionarioSpam, paramLaplaceSmoothing, palabras_spam)
    phw = dictionaryProbability(diccionarioHam, paramLaplaceSmoothing, palabras_ham)

    return probabilidadHam, probabilidadSpam


# Ham = 1
# Spam = 0
# Print data
print(bayesLaplaceSmoothingModel(training_set))
print(training_set["spam"].value_counts())
