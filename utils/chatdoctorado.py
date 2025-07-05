import os
import json
import datetime
import requests
import aiohttp
from utils.rag_search import buscar_chunks_relevantes
from utils.reranker import rerank_hibrido
from utils.respuesta_distinta import son_parecidas_llm 


class ChatDoctorado:
    def __init__(self, historial):
        self.respuestas_vacias = 0 # Creamos una variable para llevar el recuento del número de veces que el sistema da uan respuesta vacia
        self.historial = historial if historial is not None else []

        self.json_log_path = os.path.join("logs", "log_preguntas_sin_respuesta.json")
        self.log_path = os.path.join("logs", "log_preguntas_sin_respuesta.txt")
        os.makedirs("logs", exist_ok=True)

    # Creamos una función para guardar las veces que no obtiene una respuesta el usuario, debido a que no se devuelvan
    # documentos en el RAG.
    def registrar_sin_respuesta(self, pregunta, respuesta, contexto):
        """
        Guardamos el tiempo, la pregunta, respuesta y contexto en un fichero txt y en un fichero JSON.
        Aquí se guardan las preguntas que no han recibido una respuesta del modelo generativo y que por tanto,
        han recibido una respuesta por defecto.
        
        """

        # Guardamos el tiempo actual
        timestamp = datetime.datetime.now().isoformat()
        # Abrimos el fichero, guardamos la pregunta, la respuesta generada y el contexto usado.
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"\n=== {timestamp} ===\n")
            f.write(f"Pregunta: {pregunta}\n")
            f.write(f"Respuesta generada: {respuesta.strip()}\n")
            f.write(f"Contexto usado:\n{contexto.strip()}\n")

        entrada_json = {
            "timestamp": timestamp,
            "pregunta": pregunta,
            "respuesta_generada": respuesta.strip(),
            "contexto_usado": contexto.strip()
        }

        # Primero comprobamos si el archivo json para guardar los logs existe
        if not os.path.exists(self.json_log_path):
            # Si no existe, se crea y se le escribe la primera entrada pasando a formato json lo anterior
            with open(self.json_log_path, "w", encoding="utf-8") as jf:
                json.dump([entrada_json], jf, ensure_ascii=False, indent=2)
        else:
            # En caso en el que si existiera, pasamos a formato json lo anterior y lo añadimos al final 
            with open(self.json_log_path, "r+", encoding="utf-8") as jf:
                data = json.load(jf)
                data.append(entrada_json)
                # Movemos a continuación el cursor al principio del archivo para evitar problemas por haberlo
                # abierto en modo r+
                jf.seek(0)
                json.dump(data, jf, ensure_ascii=False, indent=2)


    async def obtener_respuesta_llama(self, pregunta, contexto, contexto_previo=None, modelo="llama3.2"):
        """
        Esta función se encarga de realizar la parte generativa del RAG.
        Recibe, la pregunta del usuario, el historial previo de la conversación, los chunks devueltos por el 
        re-ranker y el prompt.
        Devuelve la respuesta que se le pasará al usuario.
        
        """

        # Creamos una lista vacia para meterle los elementos que le pasaremos en el prompt.
        prompt_partes = []
        # Si hay contexto, es decir, el RAG ha devuelto algún chunk, lo añadimos.
        if contexto:
            prompt_partes.append(f"Contexto actual:\n{contexto}")
        # Si hay contexto previo, es decir, no es None, añadimos el contexto previo
        if contexto_previo:
            prompt_partes.append(f"Contexto previo resumido:\n{contexto_previo}")
        # Añadimos la pregunta
        prompt_partes.append(f"Pregunta: {pregunta}")
        # Añadimos el prompt para el modelo.
        prompt_partes.append("Responde a la pregunta del usuario basandote solo en la información relevante que hayas recibido. Si no sabes la respuesta, indícalo.")
        prompt = "\n\n".join(prompt_partes)

        data = {
            "model": modelo,
            "messages": [
                {"role": "system", "content": "Eres un asistente que ayuda con preguntas sobre doctorados."},
                {"role": "user", "content": prompt}
            ]
        }
        timeout = aiohttp.ClientTimeout(total=3000)
        # Hacemos la llamada al modelo
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post("http://localhost:11434/api/chat", json=data) as response:
                    respuesta_llama = ""
                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if not line:
                            continue
                        try:
                            json_data = json.loads(line)
                            contenido = json_data.get("message", {}).get("content", "")
                            respuesta_llama += contenido
                        except json.JSONDecodeError:
                            continue

            # Comprobamos si el modelo ha devuelto una respuesta vacia.
            if respuesta_llama.strip() == "":
                # En caso de devolverla se añade un valor a la variable, para tener un recuento de cuantas
                # veces da una respuesta por defecto porque no encuentra información.
                self.respuestas_vacias += 1

                # Guardamos la pregunta, la respuesta que en este caso es el vacio y el contexto en caso de haberlo,
                # sino guardamos un string vacio, en el fichero de sin respuesta.
                self.registrar_sin_respuesta(pregunta, "", contexto or "")

                # Devolvemos una respuesta por defecto en el que le indicamos que no tiene información.
                # Esto termina la función por lo que hace que no se ejecuta nada más posteriormente
                return "Lo siento, no tengo información para esa pregunta."

            # Verificamos similitud con la última respuesta.
            # Si hay historial entramos dentro para ver si se parece a la última respuesta dada, si no hay historial, 
            # no tiene sentido hacer esto.
            if self.historial:
                ultima_respuesta = self.historial[-1]["respuesta"]
                # Llamamos a la función que se encuentra en respuesta_distinta.py para comprobar si se parece
                # o no la respuesta dada a la respuesta anterior.
                # Esto nos dará un True o un False, en el caso de que sean parecidas o en el caso de que no
                # Si son parecidas, es decir, devuelve True, entonces entramos en el if.
                if await son_parecidas_llm(ultima_respuesta, respuesta_llama, modelo=modelo):
                    print("🔁 Respuesta parecida detectada. Solicitando una alternativa...")
                    # Añadimos una nueva instrucción al modelo para que genere la respuesta de forma distinta
                    prompt_partes.append("Por favor, da una respuesta distinta o con otro enfoque.")
                    prompt = "\n\n".join(prompt_partes)
                    data["messages"][-1]["content"] = prompt

                    # Volvemos a hacer la llamada al modelo con el nuevo prompt
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post("http://localhost:11434/api/chat", json=data) as response:
                            respuesta_llama = ""
                            async for line in response.content:
                                line = line.decode("utf-8").strip()
                                if not line:
                                    continue
                                try:
                                    json_data = json.loads(line)
                                    contenido = json_data.get("message", {}).get("content", "")
                                    respuesta_llama += contenido
                                except json.JSONDecodeError:
                                    continue
            
            # Devolvemos la respuesta del modelo.
            # Si hubo respuesta y no era parecida a una anterior, devuelve esa respuesta
            # Si era parecida devuelve la nueva
            return respuesta_llama

        except requests.exceptions.RequestException as e:
            return f"❌ Error en la llamada al modelo Llama: {e}"

    async def generar_respuesta_fallback(self, pregunta, modelo="llama3.2"):
        """
        Esta función usa un modelo de lenguaje para devolver una respuesta general al usuario en caso de
        que no se haya encontrado documentos relevantes. 
        Esta se basa en pedir al usuario que reformule su cuestión.
        
        """

        mensaje = (
            f"No se ha encontrado ningún contenido relevante para la siguiente pregunta:\n\n"
            f"{pregunta}\n\n"
            "Sugiere al usuario que reformule la pregunta con otras palabras, "
            "o que consulte directamente la web oficial de la Universidad de Sevilla. "
            "Indica que no se han encontrado documentos relevantes para ayudarle."
        )
        data = {
            "model": modelo,
            "messages": [
                {"role": "system", "content": "Eres un asistente amable y útil."},
                {"role": "user", "content": mensaje}
            ]
        }

        try:
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

            # Devuelve la respuesta del modelo sin espacios delante o detrás.
            return respuesta.strip()
        except Exception as e:
            print(f"⚠️ Error generando respuesta fallback: {e}")
            return (
                "No se ha encontrado contenido relevante. "
                "Puedes reformular tu pregunta o consultar la web oficial de la Universidad de Sevilla."
            )

    async def buscar_respuesta(self, pregunta, vectorstore, contexto_previo=None, top_k=3):
        """
        Busca los chunks relevantes para la pregunta y a partir de ellos le da la respuesta generativa al usuario.
        Recibe:
            - Pregunta del usuario
            - Vectorstore, donde estan almacenados los embedding
            - Top_k, que es el número de documentos que va a devolver en el RAG
        Devuelve un texto generado a partir de los chunks seleccionados.
        
        """
        # Llamamos a la función de rag_search.py
        # Esta función nos devuelve una tupla con la url y el texto
        chunks_raw = await buscar_chunks_relevantes(pregunta, vectorstore, top_k=top_k)
        print("\n🧾 Chunks iniciales devueltos por la búsqueda:")
        # Para cada uno de los documentos imprimimos por pantalla su posición (el primero que se devuelva será
        # el primero que hayamos guardado), la url y el texto (solo una parte para no saturar).
        # Lo de enumerate(chunks_raw,1) es para que empiece la númeración desde el 1 en vez de desde el cero
        for i, (doc_url, chunk_texto, _ ) in enumerate(chunks_raw, 1):
            print(f"[{i}] Chunk {i} ID: {doc_url}\n{chunk_texto[:300]}...\n")

        # Llamamos a la funcion de reranker.py para realizar el re-ranking de los chunks obtenidos
        chunks_rankeados = await rerank_hibrido(pregunta, chunks=[(chunk_text, embedding) for _, chunk_text, embedding in chunks_raw], umbral=0.5, verbose=True)

        # Si no se ha devuelto nada, mostramos un mensaje por pantalla que indica que ningún chunk supero el umbral
        if not chunks_rankeados:
            print("\n❌ Ningún chunk superó el umbral. Generando respuesta alternativa.")
            # usamos la función definida previamente para devolverle una respuesta generica la usuario.
            fallback = await self.generar_respuesta_fallback(pregunta)
            # Añadimos la pregunta y la repuesta al historial
            self.historial.append({
                "pregunta":pregunta, 
                "respuesta":fallback
                }) 
            # Añadimos la pregunta al registro de las sin respuesta
            self.registrar_sin_respuesta(pregunta, fallback, contexto="")
            return fallback, []
        
        #En caso de si haber chunks relevantes

        print("\n📊 Puntuaciones de relevancia asignadas por el LLM:")
        for i, (_, chunk_texto, score) in enumerate(chunks_rankeados, 1):
            print(f"[{i}] Score: {score:.2f}\n{chunk_texto[:300]}...\n")

        # Añadimos al contexto los chunks que hayamos obtenido
        contexto = "\n---\n".join(chunk_texto for _, chunk_texto, _ in chunks_rankeados)
        print("\n✅ Chunks seleccionados para el modelo:")
        print(contexto)

        # Llamamos a la función definida anteriormente para que nos de la respuesta del modelo.
        respuesta = await self.obtener_respuesta_llama(pregunta, contexto, contexto_previo=contexto_previo)

       # Añadimos la respuesta al historial junto con la pregunta
        self.historial.append({
            "pregunta":pregunta,
            "respuesta": respuesta})

        # En caso de que se devuelva "Lo siento, no tengo información" por parte del modelo generativo
        # Añadimos un log de error
        if "Lo siento, no tengo información" in respuesta:
            self.respuestas_vacias += 1
            self.registrar_sin_respuesta(pregunta, respuesta, contexto)
        else:
            self.respuestas_vacias = 0

        # Devolvemos la respuesta
        return respuesta, chunks_rankeados
