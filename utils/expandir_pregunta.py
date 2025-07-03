import requests
import json

PROMPT_EXPANSION = """
Tu tarea es ayudar a clarificar y expandir preguntas para mejorar la recuperación de información.

Dada la siguiente pregunta, si está ambigua o incompleta, devuelve solo una versión más explícita y clara que aclare posibles siglas, pronombres, abreviaciones o referencias implícitas.

Si la pregunta ya está suficientemente clara, responde con exactamente la misma pregunta.

Ejemplo:
Pregunta: ¿qué requisitos hay para un doctorado?
Respuesta: ¿qué requisitos hay para acceder a un programa de doctorado?

Pregunta: {pregunta}
Respuesta:
"""

def expandir_pregunta(pregunta: str, modelo="llama3.2") -> str:
    prompt = PROMPT_EXPANSION.format(pregunta=pregunta)

    url = "http://localhost:11434/api/chat"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": modelo,
        "messages": [
            {"role": "system", "content": "Eres un asistente que ayuda a clarificar y expandir preguntas ambiguas respondiendo solo con la pregunta expandida o cadena vacía si ya está clara."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, json=data, headers=headers, stream=True)

        if response.status_code == 200:
            respuesta_completa = ""
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        json_data = json.loads(line)
                        if "message" in json_data and "content" in json_data["message"]:
                            respuesta_completa += json_data["message"]["content"]
                    except json.JSONDecodeError:
                        pass  # Ignorar líneas que no sean JSON válido

            respuesta_final = respuesta_completa.strip()

            # Si la respuesta está vacía o igual a la pregunta original, devolver la pregunta original (sin cambios)
            if not respuesta_final or respuesta_final.lower() == pregunta.lower():
                return ""

            return respuesta_final
        else:
            print(f"❌ Error al expandir pregunta: {response.status_code}")
            return ""
    except Exception as e:
        print(f"❌ Excepción durante la expansión: {e}")
        return ""
