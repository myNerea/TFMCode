import aiohttp
import json
from utils.utils_idiomas import detectar_idioma_llm

async def traducir_a_espanol_llm(texto, modelo="llama3.2"):
    """
    Genera la traducción al español del texto recibido mediante el uso de un modelo Llama3.2.
    Recibe un texto, que será la pregunta del usuario.
    Devuelve el texto traducido.

    """

    prompt = (
        "Por favor, traduce el siguiente texto al español de forma precisa y natural. "
        "Solo devuelve la traducción, sin explicaciones, notas, ni ningún texto adicional. "
        "No incluyas comillas ni etiquetas, solo el texto traducido.\n\n"
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
        async with aiohttp.ClientSession() as session:
            async with session.post("http://localhost:11434/api/chat", json=data) as response:
                response.raise_for_status()

                traduccion = ""
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line:
                        try:
                            json_data = json.loads(line)
                            contenido = json_data.get("message", {}).get("content", "")
                            traduccion += contenido
                        except json.JSONDecodeError:
                            pass
                # Devuelve la traducción eliminando espacios que se hayan podido generar delante o detrás de la frase.
                return traduccion.strip()

    except Exception as e:
        print(f"❌ Error en traducción LLM: {e}")
        return texto


async def preparar_pregunta(texto, modelo="llama3.2"):
    """
    Detecta idioma y si hay mezcla. Si no está en español, traduce.
    Devuelve: texto_español, idioma_detectado, mezcla_idiomas
    """
    # Usamos la funcion en utils_idomas.py para determinar el idioma principal y si había mezcla.
    # Info contedrá un diccionario.
    info = await detectar_idioma_llm(texto, modelo=modelo)
    # Extraemos la clave del diccionario para obtener el idioma principal detectado.
    idioma = info.get("idioma_detectado", "español")
    # Extraemos la clave del diccionario para obtener si había o no mezcla de idiomas.
    mezcla = info.get("mezcla_idiomas", False)

    # Si el idioma es distinto de español, llamamos a la función anterior para generar la traducción mediante
    # un LLM.
    if idioma != "español":
        traduccion = await traducir_a_espanol_llm(texto, modelo=modelo)
        # Mostramos la traducción que devuelve el modelo.
        print(f"📝 Traducción pregunta: {traduccion}")
        # Mostramos el idioma original en el que estaba la pregunta.
        print(f"✅ La pregunta se detectó en idioma: {idioma}. Se tradujo a español si era distinto.")
        # Devolvemos la traduccion, el idioma principal y si el texto tenía o no mezcla de idiomas.
        return traduccion, idioma, mezcla
    else:
        # En caso de estar en español, devolvemos el texto original que se introdujo, el idioma principal y
        # si tenía mezcla o no.
        # Téngase en cuenta, que si entra aquí, el idioma es español y mezcla es False.
        print("✅ La pregunta ya está en español, no se tradujo")
        return texto, idioma, mezcla
