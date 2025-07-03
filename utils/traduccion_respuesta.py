import requests
import json
from utils.utils_idiomas import detectar_idioma_llm  # Asegúrate de que está disponible

def traducir_desde_espanol_llm(texto, idioma_destino, modelo="llama3.2"):
    """
    Traduce un texto desde español a otro idioma usando un LLM.
    Verifca que la salida este en el idioma deseado. En caso contrario salta un aviso.
    """
    prompt = (
        f"Por favor, traduce el siguiente texto del español al idioma '{idioma_destino}' de forma precisa y natural. "
        f"Asegúrate de que el resultado esté 100% en '{idioma_destino}'. "
        "NO devuelvas ningún texto en otro idioma. "
        "Solo devuelve la traducción, sin explicaciones, notas ni texto adicional. "
        "No incluyas comillas ni etiquetas, solo el texto traducido directamente.\n\n"
        f"{texto}"
    )

    data = {
        "model": modelo,
        "messages": [
            {
                "role": "system",
                "content": "Eres un traductor profesional. Solo proporcionas la traducción solicitada sin agregar nada más."
            },
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post("http://localhost:11434/api/chat", json=data, stream=True)
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

        # Devolvemos la traducción sin espacios por delante y por detrás.
        traduccion = traduccion.strip()

        # Devolvemos la traducción.
        print(f"🌍 Traducción {traduccion}")
        # Validamos que el idioma detectado coincide con el destino.
        # Para ello, llamamos a la función de detección del idioma utilizada antes.
        info_validacion = detectar_idioma_llm(traduccion)
        # Intenta obtener el valor cuya clave del diccionario es idioma_detectado, si no la encuentra, devuelve
        # la cadena vacia.
        idioma_detectado = info_validacion.get("idioma_detectado", "")

        if idioma_detectado != idioma_destino:
            print(f"⚠️ Traducción fallida: se esperaba '{idioma_destino}', pero se detectó '{idioma_detectado}'")

        # Devolvemos la traducción.
        return traduccion

    except Exception as e:
        print(f"❌ Error en traducción LLM: {e}")
        return texto


def traducir_respuesta(respuesta, idioma_destino, modelo="llama3.2"):
    """
    Llama a la función para traducir la respuesta al idioma destino si este es distinto del español.
    Si el idioma destino es 'español', no se hace nada.
    """
    respuesta = respuesta.strip()
    # Si no se ha obtenido una respuesta o el idioma es español, entonces se devuelve la respuesta
    if not respuesta or idioma_destino == "español":
        print("✅ El idioma original era el español.")
        return respuesta
    # En caso de que el idioma de destino sea distinto del español entonces llamamos al modelo para realizar
    # la traduccion
    return traducir_desde_espanol_llm(respuesta, idioma_destino, modelo=modelo)
