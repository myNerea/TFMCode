import aiohttp
import json


async def son_parecidas_llm(texto1, texto2, modelo="llama3.2", umbral=0.85):
    """
    Usa un LLM para evaluar si dos textos son demasiado parecidos.
    Devuelve True si son similares (por encima del umbral), False si no.

    Args:
        texto1 (str): Texto original.
        texto2 (str): Nuevo texto generado.
        modelo (str): Modelo a usar vía Ollama.
        umbral (float): Umbral de similitud (0-1).

    Returns:
        bool: True si son parecidas, False si no.
    """
    prompt = (
        "Dado el siguiente texto A y texto B, evalúa si el contenido de ambos es demasiado similar. "
        "Responde SOLO con un número entre 0 y 1, donde 1 significa que son prácticamente iguales en contenido, "
        "y 0 significa que son completamente distintos.\n\n"
        f"Texto A:\n{texto1}\n\n"
        f"Texto B:\n{texto2}\n\n"
        "¿Qué tan similares son A y B?"
    )

    data = {
        "model": modelo,
        "messages": [
            {"role": "system", "content": "Eres un asistente que compara dos textos y devuelve una puntuación de similitud."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        # Intentamos hacer la llamada al modelo
        async with aiohttp.ClientSession() as session:
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
        
        # Recogemos la respuesta del modelo que será un número
        score = float(respuesta.strip().replace(",", "."))
        # Devolvemos el valor correspondiente que resulte de comprobar si ese valor dado es mayor o igual que 
        # el umbral que hemos establecido.
        return score >= umbral
    except Exception as e:
        print(f"❌ Error al comparar respuestas: {e}")
        return False
