import json
import requests
import re


def limpiar_json_de_llm(respuesta_cruda):
    """
    Limpia la respuesta del LLM para eliminar delimitadores tipo ```json ... ```
    Si la respuesta es un dict, intenta limpiar el texto dentro de sus valores.
    """
    # Primero comprobamos si es un diccionario la respuesta que obtenemos
    if isinstance(respuesta_cruda, dict):
        # Intentar limpiar cadenas dentro del dict
        # Iteramos sobre las clave/valores del diccionario. Aunque para la limpieza solo tendremos en cuenta
        # el valor.
        for _ , val in respuesta_cruda.items():
            # Procesamos solo los valores que sean cadenas de texto, si no lo son, se ignoran
            if isinstance(val, str):
                # Limpiamos la cadena para quitar posibles bloques de código Rmarkdown que se hayan podido introducir
                # Primero eliminamos los espacios delante y detras de los valores
                # Después buscamos todas las coincidencias del patrón y las reemplazamos por una cadena vacia, lo que 
                # equivale a eliminar esos elementos que coincidan con la expresión regular.
                # Ignorecase ignora que las mayusculas y minusculas, es decir, las trata todas por igual
                # Multiline permite que ^ se aplique a todas las lineas, no solo a la primera
                texto_limpio = re.sub(r"^```(?:json)?\s*|\s*```$", "", val.strip(), flags=re.IGNORECASE | re.MULTILINE)
                try:
                    # Creamos un fichero json valido a partir del texto ya limpio
                    return json.loads(texto_limpio)
                except:
                    continue
        # No se pudo limpiar, devolver dict original
        return respuesta_cruda

    # En caso de que sea un string, aplicamos la limpieza directamente
    elif isinstance(respuesta_cruda, str):
        texto_limpio = re.sub(r"^```(?:json)?\s*|\s*```$", "", respuesta_cruda.strip(), flags=re.IGNORECASE | re.MULTILINE)
        return texto_limpio
    # Si no es nada de lo anterior lo devolvemos directamente
    else:
        return respuesta_cruda


def evaluar_saludo_social(input_usuario, respuesta_generada, modelo="llama3.2"):
    """
    Esta función se encarga de evaluar la actuación del modelo al responder cuando entra en el módulo social.
    Recibe el input del usuario y la respuesta generada.
    Devuelve un JSON con una serie de valores que representan lo bien o lo mal que lo ha hecho el modelo.
    
    """

    prompt = f"""
Evalúa la calidad de una respuesta social generada por un asistente conversacional.

Devuelve **únicamente** un JSON válido con los siguientes campos, sin ningún texto adicional, sin comillas, sin bloques de código ni explicaciones:

{{
  "relevancia": "alta" | "media" | "baja",
  "tono": "amable" | "neutral" | "inapropiado",
  "coherencia_semantica": "alta" | "media" | "baja",
  "comentario": "explicación breve del juicio"
}}

- Mide la relevancia en función de lo acorde que esta la respuesta del modelo a la pregunta. 
- El tono se mide con cómo de agradable/amable pueda resultar la respuesta para un ser humano.
- La coherencia semantica se mide como el grado de unidad del texto.
Los siguientes ejemplos tienen relevancia alta, tono amable y coherencia_semantica alta:
Usuario: "Hola" -> Modelo: "¡Hola! ¿En qué puedo ayudarte?"
Usuario: "Muchas gracias" -> Modelo: "De nada, para eso estoy. ¿Puedo ayudarte en algo más?"
Usuario: "nada más" -> Modelo: "Espero haberte sido de ayuda. Si tienes alguna otra cuestión no dudes en preguntarme"
Usuario: "adiós" -> Modelo: "¡Hasta luego! Que tengas un buen día."

Mensaje del usuario:
{input_usuario}

Respuesta del asistente:
{respuesta_generada}
""".strip()

    # Usamos la función que definimos abajo para llamar al modelo. Esto se hace pasandole el prompt y el modelo
    # correspondiente que vamos a usar.
    respuesta_cruda = _llamar_llm(prompt, modelo)

    # Limpiar la respuesta para eliminar bloques de código u otros textos extra
    resultado_limpio = limpiar_json_de_llm(respuesta_cruda)

    # Si es un diccionario lo devolvemos.
    if isinstance(resultado_limpio, dict):
        return resultado_limpio

    # Si no lo es intentamos convertirlo a diccionario.
    try:
        resultado = json.loads(resultado_limpio)
    except json.JSONDecodeError as e:
        # Si falla, devolver un dict de error para manejarlo luego
        resultado = {
            "error": str(e),
            "respuesta_cruda": respuesta_cruda
        }

    return resultado

# En la siguiente función definimos el prompt necesario para generar la respuesta.
def evaluar_reexplicacion(texto_original, texto_reexplicado, modelo="llama3.2"):
    """
    Esta función se encarga de evaluar la actuación del modelo al responder cuando entra en el módulo de reexplicacion.
    Recibe el input del usuario y la respuesta generada.
    Devuelve un JSON con una serie de valores que representan lo bien o lo mal que lo ha hecho el modelo.
    
    """

    prompt = f"""
Evalúa si una reexplicación cumple con estos criterios:
- Mantiene fidelidad al contenido original.
- Usa lenguaje más sencillo o accesible.
- No inventa contenido nuevo.

Devuelve este JSON:
{{
  "fidelidad": "alta" | "media" | "baja",
  "simplificacion": "alta" | "media" | "baja",
  "alucinacion": true | false,
  "comentario": "evaluación en pocas palabras"
}}

Texto original:
{texto_original}

Reexplicación generada:
{texto_reexplicado}
""".strip()

    return _llamar_llm(prompt, modelo)


def evaluar_respuesta_no_rag(pregunta, respuesta, modelo="llama3.2"):
    """
    Evalúa la calidad de la respuesta generada cuando el modelo responde sin contexto RAG.
    Devuelve únicamente un JSON con campos para evaluar relevancia, claridad y coherencia.
    """
    prompt = f"""
Evalúa la calidad de una respuesta generada por un modelo sin acceso a contexto adicional (no RAG).

Devuelve únicamente un JSON válido con estos campos:

{{
  "relevancia": "alta" | "media" | "baja",
  "claridad": "alta" | "media" | "baja",
  "coherencia": "alta" | "media" | "baja",
  "comentario": "explicación breve del juicio"
}}

Pregunta del usuario:
{pregunta}

Respuesta generada:
{respuesta}
""".strip()

    # Guardamos la respuesta del modelo
    respuesta_cruda = _llamar_llm(prompt, modelo)
    # Limpiamos la respuesta del modelo
    resultado_limpio = limpiar_json_de_llm(respuesta_cruda)

    # En caso de que sea un diccionario lo devolvemos
    if isinstance(resultado_limpio, dict):
        return resultado_limpio

    # En caso contrario intentamos ponerlo en formato JSON y si no podemos devolvemos un JSON con el 
    # error y con la respuesta del modelo
    try:
        resultado = json.loads(resultado_limpio)
    except json.JSONDecodeError as e:
        resultado = {
            "error": str(e),
            "respuesta_cruda": respuesta_cruda
        }
    return resultado


def evaluar_pregunta_clarificacion(pregunta_original, pregunta_clarificadora, modelo="llama3.2"):
    """
    Evalúa la calidad de una pregunta clarificadora generada para solicitar más información al usuario.
    Devuelve un JSON con campos para evaluar pertinencia, claridad y utilidad.
    """
    prompt = f"""
Evalúa la calidad de una pregunta clarificadora generada para solicitar más información a un usuario.

Devuelve únicamente un JSON válido con estos campos:

{{
  "pertinencia": "alta" | "media" | "baja",
  "claridad": "alta" | "media" | "baja",
  "utilidad": "alta" | "media" | "baja",
  "comentario": "explicación breve del juicio"
}}

Pregunta original:
{pregunta_original}

Pregunta clarificadora generada:
{pregunta_clarificadora}
""".strip()

    # Obtenemos la respuesta del modelo llamando a la función definida abajo
    respuesta_cruda = _llamar_llm(prompt, modelo)
    # Limpiamos la respuesta
    resultado_limpio = limpiar_json_de_llm(respuesta_cruda)

    # Si la respuesta es de tipo diccionario la devolvemos
    if isinstance(resultado_limpio, dict):
        return resultado_limpio

    # En caso contrario intentamos convertirla a objeto JSON
    try:
        resultado = json.loads(resultado_limpio)
    except json.JSONDecodeError as e:
        resultado = {
            "error": str(e),
            "respuesta_cruda": respuesta_cruda
        }
    return resultado


def _llamar_llm(prompt, modelo):
    """
    Esta función llama al modelo pasandole el prompt correspondiente. 
    Devuelve un JSON válido.
    Permite hacer un bloque reutilizable para las tres funciones anteriores.

    """

    url_api = "http://localhost:11434/api/chat"
    data = {
        "model": modelo,
        "messages": [
            {"role": "system", "content": "Devuelve solo un JSON válido según las instrucciones."},
            {"role": "user", "content": prompt}
        ],
        "stream": True
    }

    respuesta_llm = ""
    try:
        response = requests.post(url_api, json=data, stream=True)
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    json_data = json.loads(line)
                    contenido = json_data.get("message", {}).get("content", "")
                    respuesta_llm += contenido
                except json.JSONDecodeError:
                    continue
        # Devolvemos la respuesta como un JSON.
        return json.loads(respuesta_llm)
    except Exception as e:
        return {"error": str(e), "respuesta_cruda": respuesta_llm}