#Imports 
from collections import defaultdict
import re
import pandas as pd
import manejoCSV as archivo

# Open file 
archivo.TxtReaderWriter(r"entrenamiento.txt", r"entrenamiento.csv")

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
with open("training_set.csv", "w",encoding='utf-8') as training:
    training.write(training_set.to_csv(index=False))
with open("test_set.csv", "w",encoding='utf-8') as test:
    test.write(test_set.to_csv(index=False))
with open("validation_test.csv", "w",encoding='utf-8') as validation:
    validation.write(validation_test.to_csv(index=False))

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
    probabilidadSpam = (training_set["spam"].value_counts()[0] +
                        paramLaplaceSmoothing) / (len(training_set) +
                                                  paramLaplaceSmoothing * 2)

    probabilidadHam = (training_set["spam"].value_counts()[1] +
                       paramLaplaceSmoothing) / (len(training_set) +
                                                 paramLaplaceSmoothing * 2)

    psw = dictionaryProbability(diccionarioSpam, paramLaplaceSmoothing,
                                palabras_spam)
    phw = dictionaryProbability(diccionarioHam, paramLaplaceSmoothing,
                                palabras_ham)
    """
    Presente al final del entrenamiento, la métrica de desempeño sobre el subset de training y sobre el subset de
    testing.
    """
  
    return probabilidadHam, probabilidadSpam, phw, psw

# Normalization of data
training_set["message"] = training_set["message"].str.replace("[^a-zA-Z]", " ")
training_set["message"] = training_set["message"].replace(r"\s+",
                                                          " ",
                                                          regex=True)
training_set["message"] = training_set["message"].str.lower()
temp_set = training_set


probham, probspam, phw, psw = bayesLaplaceSmoothingModel(training_set)