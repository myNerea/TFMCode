import aiohttp
import json

# Lo creamos como clase para que se pueda usar en varias funciones dentro del agente decisor.
class ResumidorLlama:
    """
    Llama a un modelo para generar un resumen del historial de preguntas realizadas por el usuario.
    
    """

    def __init__(self, modelo_resumen="llama3.2", url_api="http://localhost:11434/api/chat"):
        """
        Definimos el modelo que vamos a usar y la url de la api a la cual tendremos que llamar.

        """
        self.modelo = modelo_resumen
        self.url_api = url_api

        # La siguiente función obtiene el historial previo.
    async def obtener_contexto_previo(self, historial, resumen_historial_anterior):
        """
        Función para obtener el historial previo resumido.
        
        """
        if not historial:
            # Si no hay historial previo devuelve None
            historial_resumido = ""
            print("📄 No hay historial previo relevante, se evaluará la pregunta de forma aislada.")
            return historial_resumido
        # si hay historial previo une la pregunta y la respuesta en un texto, diferenciando cada par mediante
        # el uso de saltos de línea.
        texto_historial = "\n\n".join(
            f"Tú: {preg}\nAsistente: {resp}" for preg, resp in historial
        )
        # Devuelve el texto anterior resumido usando el resumidor
        historial_resumido = await self.resumir(texto_historial,resumen_historial_anterior)
        print(f"📄 Resumen del historial del usuario: {historial_resumido}")
        return historial_resumido

    async def resumir(self, texto_largo, resumen_historial_anterior):
        """
        Generamos el resumen a partir del texto dado por el usuario.

        """
        prompt = (
            "Eres un experto en resumir conversaciones. "
            "Resume brevemente el siguiente texto manteniendo solo la información relevante:\n\n"
            f"{texto_largo}\n\n. Ayudate del resumen anterior {resumen_historial_anterior}. Resumen:"
        )

        data = {
            "model": self.modelo,
            "messages": [
                {"role": "system", "content": "Eres un asistente para resumir textos."},
                {"role": "user", "content": prompt}
            ],
            "stream": True  
        }
        # Hacemos la petición POST
        # Primero le indicamos que el cuerpo del mensaje será de tipo JSON
        headers = {"Content-Type": "application/json"}
        # Realiza la petición POST, enviando data en el cuerpo del mensaje, el cual automaticamente se convierte en objeto JSON.
        # Esto se hace indicando primero la url a la que se le va a hacer la llamada y posteriormente como será el 
        # cuerpo de la llama, lo cual ha sido definido en la línea de arriba.
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.url_api, json=data, headers=headers) as response:
                if response.status == 200:
                    resumen = ""
                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if line:
                            try:
                                parsed = json.loads(line.decode("utf-8"))
                                content = parsed.get("message", {}).get("content", "")
                                resumen += content
                            except json.JSONDecodeError as e:
                                print(f"❌ Error decodificando JSON: {e}")
                    # Devolvemos el resumen sin espacios delante o detrás.
                    return resumen.strip()
                else:
                    print(f"Error al resumir texto: {response.status_code}")
                    return ""
