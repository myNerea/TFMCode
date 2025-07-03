import os
import json
import numpy as np
from utils.embedding import obtener_embedding_ollama

# Guardamos la ruta absoluta de este archivo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# A partir de la ruta anterior hacemos la búsqueda de la carpeta index
INDEX_DIR = os.path.join(BASE_DIR, "index")
# A partir de la ruta anterior hacemos la búsqueda de el archivo vectorstore
VECTORSTORE_PATH = os.path.join(INDEX_DIR, "vectorstore.json")

def cargar_vectorstore():
    """
    Función para cargar la base de datos vectorial.
    
    """
    # Si no encuentra el fichero vectorstore, devuelve un error al usuario al que le indica que no ha encontrado
    # la base de datos vectorial y le recomienda que cargue primero el fichero que la crea.
    if not os.path.exists(VECTORSTORE_PATH):
        raise FileNotFoundError("❌ El archivo vectorstore.json no existe. Ejecuta embedding.py primero.")
    # En caso de que si encuentre la base vectorial, la abre y la devuelve como objeto json
    with open(VECTORSTORE_PATH, "r") as f:
        return json.load(f)
    

def similitud_coseno(v1, v2):
    """
    Definimos la función para obtener la similitud coseno.
    Recibe dos array, que serán los embedding del chunk y de la pregunta.
    Devuelve un número que indica la similitud existente entre ambos array. Este número se encuentra entre
    0 y 1, siendo 0 nada similares y 1 exactamente iguales.
    
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)

def buscar_chunks_relevantes(pregunta, vectorstore=None, top_k=3, modelo="mxbai-embed-large"):
    """
    Función para obtener los k chunks más cercanos de la base de datos a la pregunta en base a la 
    similitud coseno.
    
    """
    # Por defecto ponemos el vectorstore como None. Así, en caso de no encontrarlo llama a la función anterior
    # para que lo cargue.
    if vectorstore is None:
        vectorstore = cargar_vectorstore()

    # Creamos los embedding de la pregunta usando la funcion que se encuentra en embedding.py
    embedding_pregunta = np.array(obtener_embedding_ollama(pregunta, modelo)).astype("float32")
    # Creamos una lista vacia para guardar los chunks que sean similares a la pregunta
    chunks_similares = []

    # Recorremos cada url de la base de datos vectorial
    for url, url_data in vectorstore.items():
        # Recorremos cada uno de los chunks asociados a esta url
        for chunk in url_data["chunks"]:
            # Convertimos el embedding de los chunks a array de numpy
            emb = np.array(chunk["embedding"]).astype("float32")
            # Calculamos la similitud coseno
            similitud = similitud_coseno(embedding_pregunta,emb)
            # Guardamos una tupla con la similitud coseno, la url, el chunk de texto, y el embedding
            chunks_similares.append((similitud, url, chunk["texto"], chunk["embedding"]))

    # Ordenamos los chunks por similitud de mayor a menor
    chunks_similares.sort(key=lambda x: x[0], reverse=True)  # mayor similitud primero

    # Devolvemos una lista con los top k chunks más relenvates (los primeros k de la lista anterior)
    # Cada elemento se devuelve como una tupla en la que traemos el texto, la url y el embedding
    return [(url, texto, embedding) for _, url, texto, embedding in chunks_similares[:top_k]]
