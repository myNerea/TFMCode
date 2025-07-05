import aiohttp
import json

async def generar_pregunta_clarificacion(pregunta_original: str, modelo="llama3.2") -> str:
    """
    Usa un modelo de lenguaje para generar siempre una pregunta intermedia
    que ayude a aclarar o expandir la pregunta original.
    El objetivo es que el usuario aporte más información para poder responder mejor.
    Recibe una pregunta y devuelve una pregunta.

    """

    prompt = (
        "Eres un asistente que ayuda a a aclarar preguntas y a obtener información más precisa "
        "Si la pregunta original es ambigua o insuficiente para dar una respuesta precisa, "
        "formula una pregunta intermedia que ayude a entender mejor la consulta. "
        "Sino, haz preguntas que permitan expandir la consulta del usuario. "
        "Genera una única pregunta que ayude a entender mejor la siguiente pregunta."
        "No incluyas texto adicional \n\n"
        f"Pregunta original: {pregunta_original}\n"
        "Pregunta para aclarar o expandir la información:"
    )

    data = {
        "model": modelo,
        "messages": [
            {"role": "system", "content": "Eres un asistente amable, curioso, útil y preciso."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        async with aiohttp.ClientSession() as session:
                async with session.post("http://localhost:11434/api/chat", json=data) as response:
                    aclaracion = ""
                    if response.status == 200:        
                        async for line in response.content:
                            line = line.decode("utf-8").strip()
                            if line:
                                try:
                                    json_data = json.loads(line)
                                    contenido = json_data.get("message", {}).get("content", "")
                                    aclaracion += contenido
                                except json.JSONDecodeError:
                                    continue

                        return aclaracion.strip()

    except Exception as e:
        print(f"❌ Error llamando al modelo para generar pregunta de clarificación: {e}")
        # En caso de error, devuelve una pregunta genérica
        return "¿Podrías dar más detalles o especificar un poco más tu pregunta?"
