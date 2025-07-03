import requests
import json
from utils.embedding import obtener_embedding_ollama
from utils.rag_search import similitud_coseno

OLLAMA_URL = "http://localhost:11434"

def llama_call(prompt: str, model: str) -> str:
    """
    Función para realizar la llama POST al modelo.
    Recibe:
        -Modelo: Modelo que vamos a usar para generar la respuesta
        -Prompt: Prompt que le vamos a pasar al modelo
        -Stream(bool): Como queremos que nos vaya mostrando la respuesta
    Devuelve un texto.
    
    """
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Eres un traductor profesional. Solo proporcionas la traducción solicitada sin agregar nada más."
            },
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(f"{OLLAMA_URL}/api/chat", json=data, stream=True)
    response.raise_for_status()

    traduccion = ""
    for line in response.iter_lines(decode_unicode=True):
        if line:
            try:
                json_data = json.loads(line)
                contenido = json_data.get("message", {}).get("content", "")
                traduccion += contenido
            except json.JSONDecodeError:
                pass
    # Devuelve la traducción eliminando espacios que se hayan podido generar delante o detrás de la frase.
    return traduccion.strip()

def translate(text: str, source_lang: str, target_lang: str) -> str:
    """
    Devuelve la traducción del texto al idioma correspondiente.
    
    """
    prompt = (
        f"Traduce el siguiente texto del {source_lang} al {target_lang} de forma precisa y natural. "
        "Solo devuelve la traducción, sin explicaciones, notas, ni ningún texto adicional. "
        "No incluyas comillas ni etiquetas, solo el texto traducido.\n\n"
        f"{text}"
    )

    return llama_call(prompt, model="llama3.2")


def evaluar_traduccion(texto: str, respuesta: str, src_lang: str = "español", tgt_lang: str = "inglés") -> dict:
    """
    Evalúa traducciones para pregunta y respuesta
    Recibe la pregunta y la respuesta, y devuelve la similitud coseno entre los embedding de ambas. Esto nos dice
    la similitud semántica existente entre ambas.
    """
    try:
        # Traducción pregunta
        traducido_pregunta = translate(texto, src_lang, tgt_lang)
        emb_original_preg = obtener_embedding_ollama(texto)
        emb_trad_preg = obtener_embedding_ollama(traducido_pregunta)
        # Calculamos la similitud coseno
        score_pregunta = similitud_coseno(emb_original_preg,emb_trad_preg)

        # Traducción respuesta
        traducido_respuesta = translate(respuesta, src_lang, tgt_lang)
        emb_original_resp = obtener_embedding_ollama(respuesta)
        emb_trad_resp = obtener_embedding_ollama(traducido_respuesta)
        score_respuesta = similitud_coseno(emb_original_resp,emb_trad_resp)

        return {
            "pregunta_original": texto,
            "pregunta_traducida": traducido_pregunta,
            "score_pregunta": round(score_pregunta, 4),
            "respuesta_original": respuesta,
            "respuesta_traducida": traducido_respuesta,
            "score_respuesta": round(score_respuesta, 4)
        }

    except Exception as e:
        return {
            "error": str(e),
            "pregunta_original": texto,
            "respuesta_original": respuesta
        }
