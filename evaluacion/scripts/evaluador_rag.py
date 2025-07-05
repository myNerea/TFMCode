import numpy as np
import json
import aiohttp
from utils.embedding import obtener_embedding_ollama  

async def obtener_todos_los_chunks_relevantes(pregunta, vectorstore, umbral_similitud=0.75, modelo="mxbai-embed-large"):
    """
    Obtenemos todos los chunks que superan un cierto umbral sin hacer una posterior selección con re-ranking
    Recibe:
        - Pregunta del usuario
        - Vectorstore
        - Umbral de similitud que vamos a considerar
        - Modelo de embedding
    Devuelve los chunks similares que ha encontrado en base a ese umbral
    
    """
    # Realizamos el embedding de la pregunta
    emb_pregunta = np.array(await obtener_embedding_ollama(pregunta, modelo)).astype("float32")
    chunks_encontrados = []

    # Buscamos los chunks dentro de la vectorstore, le calculamos la medida de similitud que en este caso será
    # la similitud coseno, pero que podría ser otra y devolvemos los chunks encontrados, junto con su url
    for url, url_data in vectorstore.items():
        for chunk in url_data["chunks"]:
            emb = np.array(chunk["embedding"]).astype("float32")
            sim = np.dot(emb_pregunta, emb) / (np.linalg.norm(emb_pregunta) * np.linalg.norm(emb) + 1e-8)
            if sim > umbral_similitud:
                chunks_encontrados.append((url, chunk["texto"]))

    return chunks_encontrados


async def evaluar_respuesta_con_llm(pregunta, respuesta_rag, chunks_rag, chunks_reranking, chunks_relevantes, modelo="llama3.2"):
    """
    Esta función recibe los chunks usados en el rag, los usados en el reranking y los considerados relevantes
    Devuelve un JSON con la evaluación de un modelo LLM de la respuesta dada por el sistema RAG
    
    """
    url_api = "http://localhost:11434/api/chat"

    # Creamos un conjunto a partir de la url y el texto obtenido en los chunks
    set_rag = set((url, texto) for url, texto in chunks_rag)
    # Creamos un conjunto a partir de la url y el texto de los chunks considerados relevantes
    set_relevantes = set((url, texto) for url, texto in chunks_relevantes)
    # Consideramos lso chunks omitidos como los que se han devuelto pero no estaban entre los relevantes
    chunks_omitidos = list(set_relevantes - set_rag)

    # Generamos un texto uniendo cada uno de los chunks con su url y su texto y separandolo por saltos de linea
    texto_chunks_usados = "\n\n".join([f"[{url}]\n{texto}" for url, texto in chunks_rag])
    texto_chunks_omitidos = "\n\n".join([f"[{url}]\n{texto}" for url, texto in chunks_omitidos])
    texto_chunks_reranking = "\n\n".join([f"{texto}" for _, texto in chunks_reranking])

    instrucciones = """
Eres un evaluador experto de calidad de sistemas RAG. Evalúa si la respuesta generada es adecuada, basándote en los documentos usados por el sistema y los omitidos.

Responde en formato JSON con los siguientes campos:
{
  "cobertura": "buena", "parcial" o "deficiente",
  "precisión": "alta", "media" o "baja",
  "alucinacion": true o false,
  "comentario": "explicación breve sobre la evaluación",
  "respuesta_mejorada": "una mejor versión de la respuesta generada"
}
""".strip()

    prompt = f"""
{instrucciones}

Pregunta del usuario:
{pregunta}

Respuesta generada:
{respuesta_rag}

Documentos utilizados por el sistema:
{texto_chunks_reranking}

Documentos omitidos pero relevantes:
{texto_chunks_omitidos} y {texto_chunks_usados}

Recuerda: {instrucciones}
""".strip()

    data = {
        "model": modelo,
        "messages": [
            {"role": "system", "content": "Eres un evaluador de calidad para sistemas RAG. Devuelves solo un JSON válido con los campos definidos."},
            {"role": "user", "content": prompt}
        ],
        "stream": True
    }

    # Creamos la variable para guardar la respuesta del modelo
    respuesta_llm = ""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url_api, json=data) as response:
            
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line:
                        try:
                            json_data = json.loads(line)
                            contenido = json_data.get("message", {}).get("content", "")
                            # Guardamos la respuesta del modelo
                            respuesta_llm += contenido
                        except json.JSONDecodeError:
                            continue
    # Si hya un error devolvemos el codigo de error, los chunks pero no se devuelve la evaluacion            
    except Exception as e:
        return {
            "error": str(e),
            "respuesta_generada": respuesta_rag,
            "evaluacion": None,
            "chunks_rag": [{"url": url, "texto": texto} for url, texto in chunks_rag],
            "chunks_reranking": [{"url": url,"texto": texto} for url, texto in chunks_reranking],
            "chunks_relevantes": [{"url": url, "texto": texto} for url, texto in chunks_relevantes],
            "chunks_omitidos": [{"url": url, "texto": texto} for url, texto in chunks_omitidos]
        }
    # Intentamos convertirlo a formato json la respuesta
    try:
        json_match = json.loads(respuesta_llm)
    # si no se puede generamos un error que se devuelve en formato json junto con la respuesta
    except json.JSONDecodeError:
        json_match = {
            "error": "No se pudo parsear la respuesta del LLM.",
            "respuesta_cruda": respuesta_llm
        }
    # Si todo ha funcionado, devolvemos un json con la pregunta, la respuesta del rag, la evaluacion que sería un
    # JSON, los chunks encontrados en el rag, los chunks devueltos en el reranking, los chunks considerados relevantes,
    # y los chunks omitidos
    return {
        "pregunta": pregunta,
        "respuesta_generada": respuesta_rag,
        "evaluacion": json_match,
        "chunks_rag": [{"url": url, "texto": texto} for url, texto in chunks_rag],
        "chunks_reranking": [{"url": url,"texto": texto} for url, texto in chunks_reranking],
        "chunks_relevantes": [{"url": url, "texto": texto} for url, texto in chunks_relevantes],
        "chunks_omitidos": [{"url": url, "texto": texto} for url, texto in chunks_omitidos]
    }
