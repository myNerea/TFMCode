import aiohttp
import json
import asyncio
import numpy as np
from utils.embedding import obtener_embedding_ollama
from utils.rag_search import similitud_coseno


async def obtener_score_llm(pregunta, chunk, modelo_llm="llama3.2"):
    """
    Usamos un modelo LLM para determinar si los chunks que devolvemos son utiles para responder la pregunta.
    Aquí lo que tenemos en cuenta es la relación semántica entre la pregunta y los chunks devueltos, obteniendo
    para cada trozo una medida de la información que aporta para responder a la pregunta.
    
    """
    prompt = (
        "Evalúa qué tan útil es el siguiente fragmento de texto para responder la pregunta dada. "
        "La utilidad se mide como la cantidad de información que aporta para poder responder a la pregunta con ella."
        "Responde SOLO con un número entre 0 (nada útil) y 1 (totalmente útil).\n\n"
        f"Pregunta: {pregunta}\n\n"
        f"Texto:\n{chunk}\n\n"
        "¿Cómo de útil es este texto para responder la pregunta?"
    )

    data = {
        "model": modelo_llm,
        "messages": [
            {"role": "system", "content": "Eres un modelo que puntúa la utilidad de un texto para responder una pregunta."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
            async with session.post("http://localhost:11434/api/chat", json=data) as response:
                respuesta = ""
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        json_data = json.loads(line)
                        content = json_data.get("message", {}).get("content", "")
                        respuesta += content
                    except json.JSONDecodeError:
                        continue

        # Lo convertimos a número real, quitandole espacios delante y detrás y sustituyendo las comas por puntos
        valor = float(respuesta.strip().replace(",", "."))
        # Realizamos una comprobación para evitar que se pase de los limites (0 y 1)
        return max(0, min(1, valor))
    except Exception as e:
        print(f"⚠️ Error evaluando relevancia LLM: {e}")
        return 0.0


async def rerank_hibrido(pregunta, chunks, umbral=0.5, modelo_embedding="mxbai-embed-large", modelo_llm="llama3.2", verbose=False):
    """
    Reordena y filtra chunks combinando similitud por embedding + score LLM.
    
    Args:
        pregunta (str): pregunta del usuario
        chunks (list[str]): lista de textos a evaluar
        umbral (float): mínimo score promedio para incluir el chunk
        modelo_embedding (str): modelo de embedding
        modelo_llm (str): modelo LLM para relevancia contextual
        verbose (bool): si True, imprime los scores

    Returns:
        list[tuple(float, str)]: lista de tuplas (score, chunk) ordenadas por score
    """

    # Usamos la funcion de embedding.py para obtener la representación vectorial de la pregunta
    pregunta_emb = await obtener_embedding_ollama(pregunta, modelo=modelo_embedding)
    resultados = []

    # Para cada uno de los chunks vemos si trae dos o tres elementos y en función de ello devolvemos una cosa
    # u otra.
    # Esto permite modificar lo que queremos mostrar
    for item in chunks:
        if len(item) == 3:
            url, chunk_text, emb_chunk = item
        else:
            chunk_text, emb_chunk = item
            url = None  # Si no hay URL, la dejamos como None
        
        score_embedding = similitud_coseno(pregunta_emb, emb_chunk)
        score_llm = await obtener_score_llm(pregunta, chunk_text, modelo_llm)
        score_medio = (score_embedding + score_llm) / 2

    
        if verbose:
            print(f"\n🔍 Chunk:\n{chunk_text[:150]}...")
            print(f"📈 Score Embedding: {score_embedding:.2f}")
            print(f"🧠 Score LLM:       {score_llm:.2f}")
            print(f"🎯 Score Promedio:  {score_medio:.2f}")

        # Los añadimos con su url, el texto y el score medio
        resultados.append((url, chunk_text, score_medio))

        await asyncio.sleep(0.3)  # evitar sobresaturación del LLM

    # Filtrar y ordenar
    # Cogemos solo aquellos que tengan un score mayor que el umbral
    chunks_relevantes = [(url, ch, sc) for url, ch, sc in resultados if sc >= umbral]
    # Los ordenamos de forma decreciente (el mayor el primero)
    chunks_ordenados = sorted(chunks_relevantes, key=lambda x: x[1], reverse=True)

    # Si url es None, esto devuelve una tupla de la forma (None, "texto del chunk", 0.8)
    return chunks_ordenados
