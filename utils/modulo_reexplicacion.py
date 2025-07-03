import json
import aiohttp

async def modulo_reexplicacion(mensaje_modelo_anterior, modelo="llama3.2"):
    """
    Reexplica el mensaje anterior de forma más clara y comprensible para el usuario.
    Recibe un texto y devuelve un texto.
    
    """

    prompt = (
        "Por favor, reescribe el siguiente mensaje de forma más clara, sencilla y fácil de entender para el usuario. "
        "No elimines información, pero usa un lenguaje más accesible y directo.\n\n"
        f"Mensaje original:\n{mensaje_modelo_anterior}"
    )

    data = {
        "model": modelo,
        "messages": [
            {
                "role": "system",
                "content": "Eres un asistente que explica conceptos de forma clara y sencilla, sin perder precisión."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:

        explicacion = ""
        async with aiohttp.ClientSession() as session:
            async with session.post("http://localhost:11434/api/chat", json=data) as response:
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line:
                        try:
                            json_data = json.loads(line)
                            contenido = json_data.get("message", {}).get("content", "")
                            explicacion += contenido
                        except json.JSONDecodeError:
                            pass
        # Devolvemos el texto dado por el modelo quitano los espacios delante y detrás.
        return explicacion.strip()

    except Exception as e:
        print(f"❌ Error en módulo de reexplicación: {e}")
        return "Disculpe, ha ocurrido un error al intentar explicarle el mensaje. ¿Desea que lo intente de nuevo?"

