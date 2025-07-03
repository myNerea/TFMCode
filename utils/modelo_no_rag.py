import requests
import json

def responder_no_rag(pregunta, modelo="llama3.2", url_api="http://localhost:11434/api/chat"):
    """
    Damos una respuesta predefinida al usuario pero que este acorde con lo que ha preguntado usando un modelo
    de lenguaje. Esto permite personalizar la experiencia del usuario pero sin llegar a contestar su pregunta.
    
    """
    data = {
        "model": modelo,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Eres un asistente que no responde preguntas fuera de su dominio de conocimiento. "
                    "Nunca respondas la pregunta. "
                    "Responde siempre amablemente que no puedes ayudar con eso, sin importar la pregunta, "
                    "sin responder directamente a la pregunta, y sugiriendo que busque en otra fuente o que sólo estás capacitado para ayudar con temas de doctorado en la Universidad de Sevilla(US)."
                )
            },
            {
                "role": "user",
                "content": f"Pregunta: {pregunta}"
            }
        ]
    }

    response = requests.post(url_api, json=data, stream=True)
    if response.status_code == 200:
        respuesta_completa = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    json_data = json.loads(line)
                    if "message" in json_data and "content" in json_data["message"]:
                        respuesta_completa += json_data["message"]["content"]
                except json.JSONDecodeError:
                    # Ignorar líneas no JSON válidas
                    pass
        # Devolvemos la respuesta del modelo sin espacios delante o detrás.
        return respuesta_completa.strip()
    # En caso de fallo, devolvemos una respuesta por defecto.
    else:
        print(f"❌ Error llamando al modelo para generar la respuesta general: {response.status_code}")
        return "Lo siento, no puedo responder a eso."
