# Instituto Politecnico Nacional
# Escuela Superior de Computo
# Trabajo Terminal 2024-B046
# Figueroa Estrada Haziel Rafael
# Pérez Bravo Isaac Ulises
# Sotelo Ramos David Salvador

import os
import numpy as np
import util.proc_img as proc_img
from PIL import Image
from io import BytesIO
from skimage import io
from skimage import img_as_ubyte
from collections import Counter
from flask import Flask, request, jsonify, send_file
from joblib import Parallel, delayed, load
import logging

# Crear la app de Flask
app = Flask(__name__)

# Configuración básica de logging
logging.basicConfig(
    level=logging.DEBUG,  # Cambia a logging.INFO o logging.ERROR en producción
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Ruta donde están guardados los modelos
model_directory = "./modelos/"  # Cambia a la ruta donde guardaste los modelos
model_files = [f for f in os.listdir(model_directory) if f.startswith("modelo_knn_") and f.endswith(".pkl")]


# Función para cargar todos los modelos (puedes cargar con lazy loading si es necesario)
def cargar_modelos(model_directory, model_files):
    modelos = {}
    for model_file in model_files:
        model_path = os.path.join(model_directory, model_file)
        modelos[model_file] = load(model_path)
    return modelos

# Función optimizada para predecir con todos los modelos y decidir por voto mayoritario
def clasificar_por_voto(modelos, lista_datos):
    # Prioridad de las clases
    prioridad_clases = {0: 3, 1: 1, 2: 2}
    umbral_atipico = 0.0

    # Predicciones con procesamiento en paralelo
    def predecir_con_modelo(modelo, dato):
        distancias, _ = modelo.kneighbors(dato, n_neighbors=1)
        distancia_minima = distancias[0][0]
        
        if distancia_minima == umbral_atipico:
            return None  # Retornar None si es atípico
        
        return modelo.predict(dato)[0]  # Retornar la clase predicha

    # Ejecutar las predicciones en paralelo
    predicciones = Parallel(n_jobs=-1)(delayed(predecir_con_modelo)(modelo, np.array(dato).reshape(1, -1))
                                       for modelo, dato in zip(modelos.values(), lista_datos))

    # Filtrar las predicciones no válidas (None)
    predicciones = [pred for pred in predicciones if pred is not None]

    if not predicciones:
        return -1, -1

    # Contar las clases predichas
    contador = Counter(predicciones)

    # Obtener las clases más votadas
    max_votos = max(contador.values())
    clases_empate = [clase for clase, conteo in contador.items() if conteo == max_votos]

    # Resolver empate en función de la prioridad de las clases
    if len(clases_empate) == 1:
        return clases_empate[0], dict(contador)
    
    clases_empate_por_prioridad = [clase for clase in clases_empate if prioridad_clases.get(clase) == min(prioridad_clases.get(c) for c in clases_empate)]

    if len(clases_empate_por_prioridad) == 1:
        return clases_empate_por_prioridad[0], dict(contador)
    
    # Si siguen empatadas, elegir la clase más votada entre las clases de mayor prioridad
    return max(clases_empate_por_prioridad, key=lambda x: contador[x]), dict(contador)

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
        logging.debug("Recibiendo solicitud POST en /predict")

        if 'image' not in request.files:
            logging.error("No se encontró una imagen en la solicitud")
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']
        logging.info(f"Archivo recibido: {image_file.filename}")

        # Cargar la imagen
        image = io.imread(image_file.stream, as_gray=True)
        logging.debug(f"Imagen cargada con forma: {image.shape}")
        image = img_as_ubyte(image)

        # Preprocesamiento
        izq, der = proc_img.crop_img(image)
        logging.debug("Imagen recortada correctamente")

        izq_proc = proc_img.filters(izq[0])
        logging.debug("Filtro aplicado al lado izquierdo")

        der_proc = proc_img.filters(der[0])
        logging.debug("Filtro aplicado al lado derecho")

        # Dividir en bloques y clasificar
        block_izq,pos_izq = proc_img.divide_image_into_blocks(izq_proc)
        logging.debug("Bloques divididos correctamente izq")
        block_der,pos_der = proc_img.divide_image_into_blocks(der_proc)
        logging.debug("Bloques divididos correctamente der")

        predict_izq= classify_blocks(block_izq)
        logging.debug("Clasificacion de bloques izq correctamente")
        predict_der= classify_blocks(block_der)
        logging.debug("Clasificacion de bloques der correctamente")

        overlay_izq = proc_img.visualize_classified_image(izq_proc, pos_izq, predict_izq)
        logging.debug("Creacion de overlay izq correctamente")

        overlay_der = proc_img.visualize_classified_image(der_proc, pos_der, predict_der)
        logging.debug("Creacion de overlay der correctamente ")

        
        # Generar la imagen final
        result = proc_img.img_org_class(image, overlay_izq, overlay_der, izq[1], der[1])
        logging.debug("Imagen final generada")

        # Convertir el array numpy a una imagen PIL
        pil_image = Image.fromarray(result)

        # Guardar la imagen en un flujo de bytes para devolverla
        img_io = BytesIO()
        pil_image.save(img_io, 'PNG')
        img_io.seek(0)

        logging.info("Respuesta enviada con la imagen procesada")
        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        logging.exception("Error durante la ejecución del endpoint /predict")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)