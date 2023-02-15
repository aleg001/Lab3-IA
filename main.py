"""
Lab 3 - Inteligencia Artificial
Fecha de inicio: 10/02/2023
"""

# Imports
from collections import defaultdict
import csv
import re
import numpy as np
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.utils.multiclass import unique_labels


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
training_set["message"] = training_set["message"].replace(r"\s+",
                                                          " ",
                                                          regex=True)
training_set["message"] = training_set["message"].str.lower()
temp_set = training_set

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
    Pendiente:
    Recuerde dejar
    justificada su respuesta en los comentarios de su código.
    Presente al final del entrenamiento, la métrica de desempeño sobre el subset de training y sobre el subset de
    testing.
    """
  
    return probabilidadHam, probabilidadSpam, phw, psw


# Ham = 1
# Spam = 0
# Print data

probham, probspam, phw, psw = bayesLaplaceSmoothingModel(training_set)

test_set["message"] = test_set["message"].str.replace("[^a-zA-Z]", " ")
test_set["message"] = test_set["message"].replace(r"\s+",
                                                          " ",
                                                          regex=True)
test_set["message"] = test_set["message"].str.lower()


def clasificar(mensaje):
    
    #Métrica  utilizar es que basada en cada oración, se le suma la probabilidad de la palbra por la probabilidad de la bayesiana, y todo se hace una sumatoria para determinar si es mayor o menor la probabilidad de spam o ham
    #Si spam, devuleve que es spam, si es ham, devuelva es ham. El caso que sean iguales, devuelve que no se sabe.
    Pspammensaje = probspam
    Phammensaje = probham

    for palabra in mensaje.split():
        if palabra in psw:
            Pspammensaje += (Pspammensaje * psw[palabra])
          
        if palabra in phw:
            Phammensaje += (Phammensaje * phw[palabra])

    print('P(Spam|mensaje):', Pspammensaje)
    print('P(Ham|mensaje):', Phammensaje)

    if Phammensaje > Pspammensaje:
        print('Este mensaje es muy probable que sea ham')
    elif Phammensaje < Pspammensaje:
        print('Este mensaje es muy probable que sea spam')
    else:
        print('Equal proabilities, have a human classify this!')

def testear(mensaje):

    #Métrica  utilizar es que basada en cada oración, se le suma la probabilidad de la palbra por la probabilidad de la bayesiana, y todo se hace una sumatoria para determinar si es mayor o menor la probabilidad de spam o ham
    #Si spam, devuleve que es spam, si es ham, devuelva es ham. El caso que sean iguales, devuelve que no se sabe.
    Pspammensaje = probspam
    Phammensaje = probham

    for palabra in mensaje.split():
        if palabra in psw:
            Pspammensaje += (Pspammensaje * psw[palabra])
          
        if palabra in phw:
            Phammensaje += (Phammensaje * phw[palabra])
    
    if Phammensaje > Pspammensaje:
       return 'ham'
    elif Pspammensaje > Phammensaje:
       return 'spam'
    else:
       return 'no se sabe'

test_set['prediccion'] = test_set['message'].apply(testear)
temp_set['prediccion'] = temp_set['message'].apply(testear)


menu = None

while menu != 5:
  print("Bienvenido al programa de detección de spam o ham\Que quiere hacer?\n")
  menu = int(input("1. Ver simulación de spam en el set de entreamiento\n2. Ver simulación de spam en el set de test.\n3. Escribir un mensaje para detectar si es spam o ham\n4. Ver simulación usando librerias\n"))
  if menu == 1:
    correctas = 0
    total = temp_set.shape[0]
    temp_set["spam"] = temp_set["spam"].map({1: "ham", 0: "spam"})
    
    for row in temp_set.iterrows():
       
       row = row[1]
       if row['spam'] == row['prediccion']:
          correctas += 1
    print
    print('Correctas:', correctas)
    print('Incorrectes:', total - correctas)
    print('Exactitud:', correctas/total)
  elif menu == 2:
    correctas = 0
    total = test_set.shape[0]
    
    for row in test_set.iterrows():
       row = row[1]
       if row['spam'] == row['prediccion']:
          correctas += 1
    print
    print('Correctas:', correctas)
    print('Incorrectas:', total - correctas)
    print('Exactitud:', correctas/total)
  elif menu == 3:
    texto = input("Ingrese un mensaje: \n")
    clasificar(texto)
  elif menu == 4:
    #Espacio de las librerías
    data = pd.read_csv('entrenamiento.csv')
    data['spam'] = np.where(data['spam']=='spam',1, 0)
    X_train, X_test, y_train, y_test = train_test_split(data['message'], 
                                                        data['spam'], 
                                                        test_size=0.2,
                                                        random_state=1234)
    vectorizer = CountVectorizer(ngram_range=(1, 2)).fit(X_train)
    X_train_vectorized = vectorizer.transform(X_train)
    X_train_vectorized.toarray().shape
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vectorized, y_train)
    predictions = model.predict(vectorizer.transform(X_test))
    print(accuracy_score(y_test,predictions))
    print(confusion_matrix(y_test,predictions))
    #La razón que esto se ve mucho mejor a difernecia de nuestro modelo, es que aquí esta utilizando ya librerías implementados de de Naive Bayes, con Multinomial
    #A parte de esto, ya se tiene un valor alpha mucho más bajo que el nuestro que lleva unos mejores resultados, y no tiene problemas con ciertos tipos de caracteres, que los tiene que remover u otra cosa.
  elif menu == 5:
    print("Gracias por su uso :3")
  else:
    print("Ingrese un valor válido")
  
   
