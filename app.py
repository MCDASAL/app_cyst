# Instituto Politecnico Nacional
# Escuela Superior de Computo
# Trabajo Terminal 2024-B046
# Figueroa Estrada Haziel Rafael
# Pérez Bravo Isaac Ulises
# Sotelo Ramos David Salvador

import os
import joblib
from io import BytesIO
import numpy as np
import util.proc_img as proc_img
from collections import Counter
from flask import Flask, request, jsonify, send_file

# Crear la app de Flask
app = Flask(__name__)

# Ruta donde están guardados los modelos
model_directory = "./modelos/"  # Cambia a la ruta donde guardaste los modelos
model_files = [f for f in os.listdir(model_directory) if f.startswith("modelo_knn_") and f.endswith(".pkl")]


# Función para cargar todos los modelos
def cargar_modelos(model_directory, model_files):
    modelos = []
    for model_file in model_files:
        model_path = os.path.join(model_directory, model_file)
        modelos.append((model_file, joblib.load(model_path)))
         
    print(modelos)
    return modelos

# Función para predecir con todos los modelos y decidir por voto mayoritario
def clasificar_por_voto(modelos, lista_datos):
     # Prioridad de las clases
     prioridad_clases = {
     0: 3,  
     1: 1,  # Mayor prioridad
     2: 2
     }
     umbral_atipico = 0.0
     # Iterar sobre la lista de datos
     predicciones = []
     for i in range(len(lista_datos)):
          dato = np.array(lista_datos[i]).reshape(1, -1)  # Asegurarse de que los datos estén en 2D
          modelo = modelos[i][1]  # Obtener el modelo correspondiente

          # Obtener distancias a los vecinos más cercanos
          distancias, _ = modelo.kneighbors(dato, n_neighbors=1)
          distancia_minima = distancias[0][0]

          # Verificar si el dato es atípico
          if distancia_minima == umbral_atipico:
              continue  # Saltar la clasificación para este dato
            
          # Clasificar si no es atípico
          prediccion = modelo.predict(dato)
          predicciones.append(prediccion[0])  # Solo la clase predicha
     
     # print (f"len: {len(predicciones)}")
     if len(predicciones) == 0:
         return -1,-1
     # Contar las clases predichas
     contador = Counter(predicciones)
     #print(contador)

     # Obtener las clases más votadas
     max_votos = max(contador.values())
     clases_empate = [clase for clase, conteo in contador.items() if conteo == max_votos]
     
     if len(clases_empate) == 1:
         # No hay empate: devolver la clase más votada
         clasificacion_final = clases_empate[0]
     else:
         # Resolver empate en función de la prioridad de las clases
         clases_empate_por_prioridad = [clase for clase in clases_empate if prioridad_clases.get(clase) == min(prioridad_clases.get(c) for c in clases_empate)]

         # Si hay un empate de prioridad, elegir la clase más votada entre las empatadas
         if len(clases_empate_por_prioridad) == 1:
             clasificacion_final = clases_empate_por_prioridad[0]
         else:
             # Si siguen empatadas, elegir la clase más votada entre las clases de mayor prioridad
             clasificacion_final = max(clases_empate_por_prioridad, key=lambda x: contador[x])

     return clasificacion_final, dict(contador)

#Extrae características GLCM para cada bloque y los clasifica
def classify_blocks(blocks):
    predictions = []
    for block in blocks:
        glcm_features = proc_img.extract_glcm_features(block)
        # Clasificar el nuevo dato
        clasificacion, _ = clasificar_por_voto(modelos, glcm_features)  # Escalar características
        predictions.append(clasificacion)
    return predictions

modelos = cargar_modelos(model_directory, model_files)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Leer la imagen enviada en la solicitud
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']

        # Preprocesar la imagen
        izq, der = proc_img.crop_img(image_file)

        izq_proc = proc_img.filters(izq[0])
        der_proc = proc_img.filters(der[0])

        block_izq = proc_img.divide_image_into_blocks(izq_proc)
        block_der = proc_img.divide_image_into_blocks(der_proc)

        # Realizar la predicción
        predict_izq, pos_izq = classify_blocks(block_izq)
        predict_der, pos_der  = classify_blocks(block_der)

        overlay_izq = proc_img.visualize_classified_image(izq_proc, pos_izq, predict_izq)
        overlay_der = proc_img.visualize_classified_image(der_proc, pos_der, predict_der)
        
        result = proc_img.img_org_class (image_file,overlay_izq,overlay_der,izq[1],der[1])

        # Guardar la imagen en un flujo de bytes para devolverla
        img_io = BytesIO()
        result.save(img_io, 'PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
