"""
Lab 3 - Inteligencia Artificial
Fecha de inicio: 10/02/2023
"""

# Imports
import modelo as model
import messageClass as ClassText
import librerias as libs



model.test_set["message"] = model.test_set["message"].str.replace("[^a-zA-Z]", " ")
model.test_set["message"] = model.test_set["message"].replace(r"\s+"," ",regex=True)
model.test_set["message"] = model.test_set["message"].str.lower()


model.test_set['prediccion'] = model.test_set['message'].apply(ClassText.testear)
model.temp_set['prediccion'] = model.temp_set['message'].apply(ClassText.testear)

menu = None

while menu != 6:
  print("\nBienvenido al programa de detección de spam o ham\Que quiere hacer?\n")
  print("1. Ver simulación de spam en el set de entreamiento\n2. Ver simulación de spam en el set de test.\n3. Escribir un mensaje para detectar si es spam o ham\n4. Ver simulación usando librerias\n5. Metrica de desempeño del modelo\n6. Salir de la simulación\n")
  menu = int(input())
  
  if menu == 1:
    correctas = 0
    total = model.temp_set.shape[0]
    model.temp_set["spam"] = model.temp_set["spam"].map({1: "ham", 0: "spam"})
    
    for row in model.temp_set.iterrows():
       
       row = row[1]
       if row['spam'] == row['prediccion']:
          correctas += 1
    print
    print('Correctas:', correctas)
    print('Incorrectes:', total - correctas)
    print('Exactitud:', correctas/total)

  elif menu == 2:
    correctas = 0
    total = model.test_set.shape[0]
    
    for row in model.test_set.iterrows():
       row = row[1]
       if row['spam'] == row['prediccion']:
          correctas += 1
    print
    print('Correctas:', correctas)
    print('Incorrectas:', total - correctas)
    print('Exactitud:', correctas/total)
    
  elif menu == 3:
    texto = input("Ingrese un mensaje: \n")
    ClassText.clasificar(texto)
    
  elif menu == 4:
    libs.useLib()
    '''
    -- RESPUESTA A LAS PREGUNTAS BRINDADAS--
    La razón principal por la cual se obtuvo un mejor resultado en la clasificación por medio de uso de librerias radica en que 
    en esta se hace uso de una distribución Multinomial mediante el uso de Naive Baye. Cabe destacar, que esta diferencia radica 
    en que en las librerias el valor  del testsize (alpha) es menor en comparacion al modelo realizado. Además, en este se aceptan 
    caracteres que en nuestro modelo no son aceptados lo cual no produce problemas o no se ve forzado a eliminar caracteres. 
    '''
  elif menu == 5:
    training_set_accuracy = model.random_data[: model.training_test_index].reset_index(drop=True)
    test_set_accuracy =  model.random_data[ model.training_test_index:].reset_index(drop=True)

    print("\nMétrica de desempeño del modelo")
    model.accuracy(training_set_accuracy,test_set_accuracy)


  elif menu == 6:
    print("Gracias por su uso :3")

  else:
    print("Ingrese un valor válido")
  
   