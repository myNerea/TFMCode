import aiohttp
import json

async def modulo_social(texto_usuario, modelo="llama3.2"):
    """
    Mediante un modelo Llama3.2 damos una respuesta concreta a cada una de las fórmulas de cortesía que el
    usuario pueda plantear.
    Este módulo recibe un texto (frase del usuario) y devuelve un texto (respuesta dada por el modelo).
    
    """

    prompt = f"""
Eres un asistente que detecta si el usuario está:
- saludando (por ejemplo: "hola", "buenos días", "buenas tardes", "buenas noches"),
- despidiéndose (por ejemplo: "adiós", "hasta luego", "nos vemos"),
- o dando las gracias (por ejemplo: "gracias", "muchas gracias", "nada más").

Según el caso, responde básandote en lo siguiente:

- Si el usuario saluda, responde: "¡Hola! ¿En qué puedo ayudarte?"
- Si el usuario se despide, responde: "¡Hasta luego! Que tengas un buen día."
- Si el usuario da las gracias: "De nada, para eso estoy. ¿Puedo ayudarte en algo más?"
- Si indica que ha terminado: "Espero haberte sido de ayuda. Si tienes alguna otra cuestión no dudes en preguntarme"
- Si no detectas ninguna de estas categorías, responde: "No estoy seguro de cómo responder a eso."

Ejemplos:
Usuario: hola
Respuesta: ¡Hola! ¿En qué puedo ayudarte?

Usuario: muchas gracias
Respuesta: De nada, para eso estoy. ¿Puedo ayudarte en algo más?

Usuario: nada más
Respuesta: Espero haberte sido de ayuda. Si tienes alguna otra cuestión no dudes en preguntarme

Usuario: adiós
Respuesta: ¡Hasta luego! Que tengas un buen día.

Usuario: buenos días
Respuesta: Buenos días, ¿En qué puedo ayudarte?

Usuario: buenas tardes
Respuesta: Buenas tardes, ¿En qué puedo ayudarte?

Usuario: buenas noches
Respuesta: Buenas noches, ¿En qué puedo ayudarte?

Usuario: {texto_usuario}
Respuesta:
""".strip()

    data = {
        "model": modelo,
        "messages": [
            {
                "role": "system",
                "content": "Eres un asistente social amable y claro."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": True
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post("http://localhost:11434/api/chat", json=data) as response:
                if response.status == 200:
                    respuesta_completa = ""
                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if line:
                            try:
                                json_data = json.loads(line)
                                contenido = json_data.get("message", {}).get("content", "")
                                respuesta_completa += contenido
                            except json.JSONDecodeError:
                                pass
                    # Devolvemos la respuesta dada por el modelo sin espacios delante o detrás.
                    texto_respuesta = respuesta_completa.strip()
                    return texto_respuesta
                else:
                    print(f"⚠️ Error en módulo social: código {response.status}")

    except Exception as e:
        print(f"Error en módulo social: {e}")
        return "Lo siento, no puedo procesar tu mensaje ahora."
