import aiohttp
import json
import re

async def detectar_idioma_llm(texto, modelo="llama3.2"):
    """
    Mediante un modelo Llama3.2 determina el idioma en el que principal en el que venía un texto y si este 
    contenía o no una mezlca de idiomas.

    Recibe un texto.
    Devuelve un JSON con dos claves:
      idioma_detectado: idioma principal del texto.
      mezcla: si había más de un idioma.
    
    """

    prompt = f"""
Analiza el siguiente texto y determina el idioma principal en que está escrito, además de si contiene mezcla clara de idiomas.

Devuelve SOLO un JSON EXACTO con esta estructura:

{{
  "idioma_detectado": "idioma principal en español, por ejemplo: español, ingles, frances, aleman, portugues, italiano",
  "mezcla_idiomas": true o false
}}

Criterios:
- "idioma_detectado": es el idioma predominante del texto. Si hay mezcla de idiomas, selecciona el que predomine gramatical o semánticamente.
- "mezcla_idiomas": true si hay dos o más idiomas claramente diferentes (por ejemplo, texto en inglés con frases en francés o español).
- Si hay mezcla, selecciona el idioma predominante (por longitud y estructura).
- Si no hay mezcla o es mínima, mezcla_idiomas debe ser false.

Ejemplos:
Texto: "Hola, ¿cómo estás? Quiero saber cómo acceder al doctorado"
→ idioma_detectado: "español", mezcla_idiomas: false

Texto: "Bonjour, je voudrais savoir how to apply al doctorado"
→ idioma_detectado: "frances", mezcla_idiomas: true

Texto: "Guten Morgen, quiero saber requisitos para el PhD"
→ idioma_detectado: "español", mezcla_idiomas: true

Texto a analizar:
\"\"\"{texto}\"\"\"
""".strip()

# \"\"\"{texto}\"\"\" Es un string multilinea que usamos por si el texto es muy extenso.
    data = {
        "model": modelo,
        "messages": [
            {
                "role": "system",
                "content": "Eres un clasificador de idiomas. Devuelves SOLO un JSON válido con los campos 'idioma_detectado' y 'mezcla_idiomas'."
            },
            {"role": "user", "content": prompt}
        ]
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post("http://localhost:11434/api/chat", json=data) as response:
                response.raise_for_status()
                # Lanza un error y evita que el resto se ejecuta si la petición obtuvo algún código de error

                contenido_completo = ""

                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        json_data = json.loads(line)
                        contenido = json_data.get("message", {}).get("content", "")
                        contenido_completo += contenido
                    except json.JSONDecodeError:
                        continue

        # Extraer el JSON usando regex
        # Esto nos devuelve el JSON que encuentre, ya que el modelo podía devolver más texto aparte, 
        # pero a nosotros solo nos interesa el JSON. Busca que contenga idioma_detectado
        # el re.DOTAIL permite que el . en la expresión regular coincida también con saltos de línea.
        # re.search devuelve un objeto Match con la primera coincidencia de la expresión regular en la cadena 
        # de texto contenido_completo. En caso de no haber coincidencias devuelve None.
        json_match = re.search(r"{\s*\"idioma_detectado\".*?}", contenido_completo, re.DOTALL)

        # Si devuelve algo lo anterior entonces not json_match es False porque un objeto Mathc es considerado
        # truthy (existe y no es vacio)
        # En caso de que lo anterior devolviese None (no ha encontrado JSON) entonces not json_match devuelve True
        # y entra
        if not json_match:
            # Genera un error que indica que el modelo de LLM no nos devolvió un JSON válido
            raise ValueError("❌ No se encontró un JSON válido en la respuesta del LLM.")

        # Usamos group(0) para devolver el texto completo sin espacios delante o detrás.
        contenido_json = json_match.group(0).strip()
        # Convertimos el texto a un JSON válido
        resultado = json.loads(contenido_json)

        # Buscamos la clave mezcla_idiomas en el diccionario. En caso de estar devuelve el valor que tiene
        # asociado, en caso de no estar, devuelve False
        mezcla = bool(resultado.get("mezcla_idiomas", False))
        # Buscamos la clave idioma_detectado en el diccionario. En caso de estar devuelve el valor que tiene
        # asociado, en caso de no estar, devuelve español
        # Además, lo convierte todo a minusculas
        idioma = resultado.get("idioma_detectado", "español").lower()

        # Si mezcla es True, forzar que el idioma por defecto sea "ingles"
        if mezcla:
            print(f"✅ La pregunta se detecto en {idioma}. Sin embargo, contenía mezcla de idiomas por lo que determino el idioma como inglés.")
            idioma = "ingles"

        # Devolvemos un diccionario que contiene el idioma detectado y si es mezcla o no, en un formato
        # limpio para poder usarlo posteriormente.
        return {
            "idioma_detectado": idioma,
            "mezcla_idiomas": mezcla
        }

    # Genera un error si no se puede entrar en el modelo LLM.
    except Exception as e:
        print(f"❌ Error en detección de idioma LLM: {e}")
        return {
            "idioma_detectado": "ingles",  # por seguridad asumimos inglés si falla
            "mezcla_idiomas": True
        }
