# Instituto Politecnico Nacional
# Escuela Superior de Computo
# Trabajo Terminal 2024-B046
# Figueroa Estrada Haziel Rafael
# Pérez Bravo Isaac Ulises
# Sotelo Ramos David Salvador

import cv2
import math
import joblib
import numpy as np
from skimage.morphology import disk, dilation
from skimage import io
from skimage import img_as_ubyte
from skimage.feature import graycomatrix, graycoprops

#funcion para recortar el area del riñon
def crop_img(image):
    # Definimos el radio de engrosamiento
    footprint = disk(3)
    
    # Convertimos la imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Usamos un filtro gaussiano
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Dilatamos los bordes BLANCOS encontrados en la imagen al radio previamente definido
    dilated = dilation(blurred_image, footprint)
    
    # Encontramos los contornos
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    largest_contour = None
    largest_area = 0
    
    # Recorre cada contorno
    for contour in contours:
        # Calcula el área del contorno
        area = cv2.contourArea(contour)
        # Si el área es mayor que el área más grande hasta ahora, actualiza las variables
        if area > largest_area:
            largest_contour = contour
            largest_area = area
    
    # Encuentra el centro del contorno más grande 
    M = cv2.moments(largest_contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    # Encuentra el borde izquierdo y derecho
    leftmost_point = min(largest_contour[:, 0], key=lambda x: x[0])
    rightmost_point = max(largest_contour[:, 0], key=lambda x: x[0])
    # Encuentra el borde inferior más cercano
    bottommost_point = min(largest_contour[:, 0], key=lambda x: x[1])
    
    # Distancia al borde izquierdo, derecho y más cercano
    distance_left = abs(leftmost_point[0] - cx)
    distance_right = abs(rightmost_point[0] - cx)
    distance_bottom = abs(bottommost_point[1] - cy)
    
    # Calculamos los centroides a partir de los cuales cortaremos la imagen
    center_kidney_left = (cx - math.ceil(distance_left / 2), cy + math.ceil(distance_bottom / 5))
    center_kidney_right = (cx + math.ceil(distance_right / 2), cy + math.ceil(distance_bottom / 5))
    
    # Obtener el tamaño de la imagen
    altura, anchura, canales = image.shape
    
    # Definimos el tamaño del recorte dependiendo de la medida más grande de la imagen
    crop_size = max(int(altura * 0.3), int(anchura * 0.3))
    
    # Riñon izquierdo
    xl1 = center_kidney_left[0] - crop_size // 2
    yl1 = center_kidney_left[1] - crop_size // 2
    xl2 = xl1 + crop_size
    yl2 = yl1 + crop_size
    cropped_image_l = image[yl1:yl2, xl1:xl2]
    
    # Riñon derecho
    xr1 = center_kidney_right[0] - crop_size // 2
    yr1 = center_kidney_right[1] - crop_size // 2
    xr2 = xr1 + crop_size
    yr2 = yr1 + crop_size
    cropped_image_r = image[yr1:yr2, xr1:xr2]

    # Coordenadas de los recortes
    coords_l = (xl1, yl1, xl2, yl2)
    coords_r = (xr1, yr1, xr2, yr2)
    
    return (cropped_image_l, coords_l), (cropped_image_r, coords_r)

def redifinir_valor_angulo(angulo):
     if angulo == 0:
          return angulo

     angulos = {0.7853981633974483: 1,
                1.5707963267948966: 2,
                2.356194490192345: 3
               }
     
     if angulos.get(angulo):
          return  angulos.get(angulo)
     
def cargar_scaler(ruta_scaler):
    try:
        # Cargar el scaler con joblib
        scaler = joblib.load(ruta_scaler)
        return scaler
    except Exception as e:
        print(f"Error al cargar el scaler desde {ruta_scaler}: {e}")
        return None

def filters (image):
     image = img_as_ubyte(image)  # Convertir la imagen a uint8
     # Paso 1: Ecualización del histograma
     equalized_image = cv2.equalizeHist(image)
     # Paso 2: Aplicar filtro Median
     imagen_procesada = cv2.medianBlur(equalized_image, 5)

     return imagen_procesada

def divide_image_into_blocks(image, block_size = 3):
    """
    Divide la imagen en bloques de tamaño block_size x block_size.
    """
    h, w = image.shape
    blocks = []
    positions = []

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # Obtener bloque
            block = image[i:i+block_size, j:j+block_size]
            if block.shape[0] == block_size and block.shape[1] == block_size:
                blocks.append(block)
                positions.append((i, j))  # Guardar la posición del bloque

    return blocks, positions

# Función para extraer características GLCM (como en tu tabla)
def extract_glcm_features(image):
     distances=[1, 2]
     angles = [0, np.pi/4, np.pi/2, np.pi*3/4]

     feature_lists = []  # Lista para almacenar las características
     
     for distance in distances:
         for angle in angles:
             # Calcular la matriz GLCM para cada combinación
             glcm = graycomatrix(image, distances=[distance], angles=[angle], levels=256, symmetric=True, normed=True)
             # Extraer las propiedades de la GLCM
             features = {
                 'correlation': graycoprops(glcm, 'correlation')[0, 0],
                 'contrast': graycoprops(glcm, 'contrast')[0, 0],
                 'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
                 'energy': graycoprops(glcm, 'energy')[0, 0]
             }

             angulo = redifinir_valor_angulo(angle)
             scaler_path = f"./scalers/scaler_dist_{distance}_ang_{angulo}.pkl"
             # Cargar el scaler correspondiente
             scaler = cargar_scaler(scaler_path)

             # Convertir las características a una lista
             feature_values = list(features.values())

             # Aplicar el scaler a las características
             feature_scaled = scaler.transform([feature_values])[0]

             # Añadir las características como una sublista
             feature_lists.append(feature_scaled)
     
     return feature_lists

def visualize_classified_image(image, positions, predictions, block_size = 3):
     #La funcion cv.rectangle utiliza el formato para colores BGR
     class_colors = {0: (0, 0, 255),  # Otros organos rojo
                    1: (0, 255, 0),  # Quiste verde
                    2: (255, 0, 0)  # Riñon azul
                    }

     # Crear una copia de la imagen original para superponer resultados
     overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
     
     for pos, pred in zip(positions, predictions):
         if pred == -1:
             continue
         x, y = pos
         color = class_colors[pred]  # Color asignado a la clase

         cv2.rectangle(overlay, (y, x), (y + block_size, x + block_size), color, -1)

     return overlay

def img_org_class (img_org,kidne_left,Kidney_right,coords_left,coords_right):
     # Creamos una copia de la imagen original para evitar modificarla directamente
     combined_image = img_org.copy()
     
     # Insertamos el recorte izquierdo tratado en las coordenadas correspondientes
     xl1, yl1, xl2, yl2 = coords_left
     combined_image[yl1:yl2, xl1:xl2] = cv2.resize(kidne_left, (xl2 - xl1, yl2 - yl1))
     
     # Insertamos el recorte derecho tratado en las coordenadas correspondientes
     xr1, yr1, xr2, yr2 = coords_right
     combined_image[yr1:yr2, xr1:xr2] = cv2.resize(coords_right, (xr2 - xr1, yr2 - yr1))
     
     # Retornamos la imagen combinada
     return combined_image