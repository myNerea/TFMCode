import requests
import json
import os
from utils.embedding import obtener_embedding_ollama
from utils.modelo_no_rag import responder_no_rag
from utils.rag_search import cargar_vectorstore, buscar_chunks_relevantes
from utils.chatdoctorado import ChatDoctorado
from utils.resumidor import ResumidorLlama
from utils.traduccion_pregunta import preparar_pregunta
from utils.traduccion_respuesta import traducir_respuesta
from utils.expandir_pregunta import expandir_pregunta
from utils.clarificador import generar_pregunta_clarificacion
from utils.social import modulo_social
from utils.modulo_reexplicacion import modulo_reexplicacion
from utils.prevencion import analizar_actitud_maliciosa, generar_respuesta_segura
from evaluacion.scripts.evaluador_traduccion import evaluar_traduccion
from evaluacion.scripts.evaluador_rag import evaluar_respuesta_con_llm, obtener_todos_los_chunks_relevantes
from evaluacion.scripts.evaluador_generativo import evaluar_saludo_social, evaluar_reexplicacion, evaluar_pregunta_clarificacion, evaluar_respuesta_no_rag
from evaluacion.scripts.latencia import Cronometro 




class AgenteDecisor:
    def __init__(self, umbral=0.5, top_k=3):
        self.vectorstore = cargar_vectorstore()
        self.resumidor = ResumidorLlama() # Creamos una instancia de la clase resumidor. 
        # Guardandola como atributo de la clase actual para poder llamarla posteriormente.
        self.historial_conversacion = [] # Aquí iremos guardando el historial de la conversación.
        self.historial_preguntas_usuario = [] # Aquí vamos guardando las preguntas que va haciendo el usuario para tenerlas como contexto.
        self.historial_resumen = ""
        self.chat_rag = ChatDoctorado(self.historial_conversacion) # Creamos una instancia de la clase chatDoctorado 
        self.umbral = umbral
        self.top_k = top_k
        self.esperando_aclaracion = False
        self.ultima_pregunta_original = None

    # Función para determinar si entrar o no en el RAG
    def es_pregunta_relevante_llm(self, pregunta, resumen_historial, modelo="llama3.2"):
        """
        Consulta al modelo para determinar si debe entrar en módulo de RAG.
        El modelo responde sí/no a si la pregunta o comentario plantea algo relacionado con el doctorado.
        """
        
        url_api = "http://localhost:11434/api/chat"

        prompt = (
            "Tu tarea es decidir si la siguiente pregunta está relacionada con temas de doctorado en la Universidad de Sevilla (US). "
            "Esto incluye requisitos, becas, procedimientos, plazos, admisiones, matrícula, normativa, equivalencias, documentación, etc. "
            "Se te proporciona un resumen de la conversación anterior para que lo tengas en cuenta en tu respuesta.\n\n"
            f"Resumen del historial:\n{resumen_historial}\n\n"
            f"Pregunta actual: {pregunta}\n\n"
            "¿Está esta pregunta relacionada con temas de doctorado en la US?" 
            "Responde solo con 'sí' o 'no'."
        )

        data = {
            "model": modelo,
            "messages": [
                {
                    "role": "system",
                    "content": "Eres un asistente que determina si una pregunta esta relacionada con doctorados en la Universidad de Sevilla, usando como apoyo en el resumen de la conversación."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": True
        }

        try:
            # Llamamos al modelo en la URL definida anteriormente y le pasamos los datos(el prompt completo)
            # Stream=True permite que se vaya leyendo tal cual va llegando por líneas o por chunks
            response = requests.post(url_api, json=data, stream=True)
            if response.status_code == 200:
                # Si obtenemos una respuesta satisfactora, la leemos
                # Esta lectura se realiza línea a línea para evitar errores de formato JSON
                respuesta_completa = ""
                for line in response.iter_lines(decode_unicode=True):
                    # decode_unicode=True decodifica cada línea de bytes a string (UTF-8)
                    if line:
                        try:
                            json_data = json.loads(line)
                            # Cada línea se intenta interpretar como JSON.
                            content = json_data.get("message", {}).get("content", "")
                            # Dentro del mensaje se extrae el contenido para guardarlo
                            respuesta_completa += content
                        except json.JSONDecodeError:
                            pass
                respuesta = respuesta_completa.strip().lower()
                # Obtenmos las respuesta limpia(sin espacios delante o atrás y en minusculas)
                return respuesta.startswith("sí") or respuesta.startswith("si")
                # Devuelve True si la respuesta del modelo empieza por sí o por si y devuelve 
                # False en caso contrario.
            else:
                print(f"⚠️ Error en API LLM: código {response.status_code}")
        except Exception as e:
            print(f"⚠️ Error al consultar LLM para relevancia con historial: {e}")

        return False
    
    # Función para determinar si es un saludo, despedida, o agradecimiento puro
    # Nota: El funcionamiento es igual que el anterior.
    def es_saludo_social_llm(self, texto, modelo="llama3.2"):
        """
        Llama al LLM para detectar si el texto es UNICAMENTE es un saludo, despedida o agradecimiento.
        El modelo devuelve sí/no en función de si lo es o no.
        Tras una comprobación, la función devuelve True si sí es un saludo, despedida o agradecimiento
        y devuelve False si no lo es.
        """
        url_api = "http://localhost:11434/api/chat"

        prompt = f"""
Eres un asistente que detecta si el usuario está saludando, despidiéndose o dando las gracias.
Responde SOLO con "sí" si el texto es un saludo, despedida o agradecimiento completamente independiente (p.ej. solo "hola", "gracias", "adiós").
Responde SOLO con "no" si el texto es más complejo, contiene más información o pregunta.

Texto: "{texto}"
¿Es este texto únicamente un saludo, despedida o agradecimiento? Responde solo con "sí" o "no".
""".strip()

        data = {
            "model": modelo,
            "messages": [
                {"role": "system", "content": "Eres un asistente que detecta saludos, despedidas o agradecimientos puros."},
                {"role": "user", "content": prompt}
            ],
            "stream": True
        }

        try:
            response = requests.post(url_api, json=data, stream=True)
            if response.status_code == 200:
                respuesta_completa = ""
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        try:
                            json_data = json.loads(line)
                            content = json_data.get("message", {}).get("content", "")
                            respuesta_completa += content
                        except json.JSONDecodeError:
                            pass
                respuesta = respuesta_completa.strip().lower()
                return respuesta.startswith("sí") or respuesta.startswith("si")
            else:
                print(f"⚠️ Error en API LLM: código {response.status_code} en saludo social")
        except Exception as e:
            print(f"⚠️ Error al consultar LLM para saludo social: {e}")

        return False

    # Función para determinar si es necesaria una reexplicación
    # Nota: El funcionamiento es igual que el anterior
    def debe_reexplicar_llm(self, pregunta_usuario, modelo="llama3.2"):
        """
        Consulta al modelo para determinar si debe entrar en módulo de reexplicación.
        El modelo responde sí/no a si la pregunta o comentario indica que no ha entendido la respuesta previa.
        """
        url_api = "http://localhost:11434/api/chat"
        prompt = (
            "Eres un asistente que debe decidir si el usuario no ha comprendido la respuesta previa y necesita que se le explique de forma más clara.\n"
            "Devuelve SOLO 'sí' o 'no'.\n\n"
            "Ejemplo:" 
            "Usuario: No lo he entendido -> sí" 
            "Usuario: No he comprendido muy bien lo que me has dicho -> sí"
            "Usuario: ¿qué requisitos hay para acceder a un programa de doctorado? -> no"
            f"Usuario: {pregunta_usuario}\n\n"
            "¿Debe el asistente intentar reexplicar la última respuesta?"
        )
        data = {
            "model": modelo,
            "messages": [
                {"role": "system", "content": "Decide si la pregunta indica falta de comprensión y necesita reexplicación."},
                {"role": "user", "content": prompt}
            ],
            "stream": True
        }

        try:
            response = requests.post(url_api, json=data, stream=True)
            if response.status_code == 200:
                respuesta_completa = ""
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        try:
                            json_data = json.loads(line)
                            content = json_data.get("message", {}).get("content", "")
                            respuesta_completa += content
                        except json.JSONDecodeError:
                            pass
                respuesta = respuesta_completa.strip().lower()
                return respuesta.startswith("sí") or respuesta.startswith("si")
            else:
                print(f"⚠️ Error en API LLM para decidir reexplicación: código {response.status_code}")
        except Exception as e:
            print(f"⚠️ Error al consultar LLM para decidir reexplicación: {e}")

        return False


    # Función que genera la respuesta para el usuario
    def responder(self, pregunta_usuario):
        """
        Esta función se encarga de generar la respuesta que recibirá el usuario.
        """

        # Creamos una nueva instancia de la clase Cronometro definida en latencia.py
        # Al crear esta nueva instancia iniciamos el cronometro que nos determinará el tiempo que se tarda
        # en dar una respuesta al usuario.
        cronometro = Cronometro()
        # Guardamos a continuación la pregunta del usuario
        cronometro.set_pregunta(pregunta_usuario)

        # Iniciamos el context manager para medir el tiempo que tarda en el módulo de traducción.
        with cronometro.medir("traduccion"):
            # Llamamos a la función para preparar la pregunta que se encuentra en traduccion_pregunta.py
            pregunta_traducida, idioma_original, mezcla = preparar_pregunta(pregunta_usuario)

        # Salimos del contexto manager creado para traduccion al terminar la indentacion por lo que se guarda
        # el tiempo que hemos tardado en traducirlo.
        # Si ocurre un error o una excepción para también el tiempo.

        # Definimos una variable que necesitaremos en el siguiente bloque with
        respuesta_final = None

        # Entramos en un nuevo bloque with. Esta vez mediremos el tiempo para el módulo de actitud maliciosa
        with cronometro.medir("analisis_actitud_maliciosa"):
            # Llamamos a la función de prevencion.py para ver si el usuario tiene o no una actitud maliciosa
            # En caso de ser True la respuesta de la función entra dentro del bloque if
            if analizar_actitud_maliciosa(pregunta_traducida):
                print("⚠️ Se detectó intento de manipulación, se genera respuesta segura.")
                # Entramos en otro with para medir el tiempo que tarda en generar la respuesta segura
                # Téngase en cuenta, que del primero no se sale hasta que no termine su bloque, el cual termina
                # cuando deja de estar indentado, llega a un returno, break o error.
                with cronometro.medir("generar_respuesta_segura"):
                    respuesta_segura = generar_respuesta_segura()

                    # Comprobamos si no era español y no tenía mezcla
                    if idioma_original != "español":
                        # En este caso nos vamos a la funcion de traduccion_respuesta.py
                        respuesta_final = traducir_respuesta(respuesta_segura, idioma_original)
                    else:
                        respuesta_final = respuesta_segura

        # Guardamos el JSON generado. Esto para automáticamente los dos with recolectando su tiempo
        cronometro.guardar_json()

        # Comprobamos si ha entrado en el bloque para generar una respuesta segura en cuyo caso devolvemos
        # el valor predefinido.
        # Tras hacer el return, se sale de la función y no ejecuta nada más.
        if respuesta_final is not None:
            return {
                "respuesta": respuesta_final,
                "necesita_aclaracion": self.esperando_aclaracion
            }

        # Definimos la variable que necesitaremos en los siguientes bloques
        respuesta_es = None


        # Comprobamos cuánto se tarda en determinar si se debe pasar o no a comprobar si usar RAG.
        # Para ello, iniciamos el cronometro de los dos módulos.
        with cronometro.medir("saludo_social_o_reexplicacion"):
            # En el caso de que la función decisora (definida en este archivo) determine que la pregunta del
            # usuario es un fórmula de cortesía, entonces entramos aquí.
            if self.es_saludo_social_llm(pregunta_traducida):
                # medimos cuanto tiempo se tarda en dar la respuesta y en generar su evaluacion en el módulo
                # social.
                with cronometro.medir("saludo_social"):
                    # Usamos la función que se encuentra en social.py
                    # Esta función nos devuelve la respuesta del modelo social (que es un texto).
                    respuesta_es = modulo_social(pregunta_traducida)
                    print("✅ Módulo social generó respuesta.")
                    # Iniciamos la evaluación de la respuesta dada por el modelo.
                    try:
                        # Medimos el tiempo que se tarda en dar una evaluación
                        with cronometro.medir("evaluacion_saludo"):
                            # Usamos la función que se puede encontrar en evaluador_generativo.py
                            # Esto obtiene un JSON con los valores de la evaluacion
                            evaluacion_social = evaluar_saludo_social(pregunta_traducida, respuesta_es)
                            print("\n🤝 Evaluación de saludo social:")
                            # Imprimimos el json
                            print(json.dumps(evaluacion_social, indent=2, ensure_ascii=False))

                            # Guardamos en un fichero la evalución
                            os.makedirs("evaluacion/json", exist_ok=True)
                            with open("evaluacion/json/saludos.json", "a", encoding="utf-8") as f:
                                # Convertimos a un JSON válido los valores
                                # Nótese, que será un diccionario que contenga a otro diccionario porque evaluacion_social
                                # era un diccionario.
                                json.dump({
                                    "pregunta": pregunta_traducida,
                                    "respuesta": respuesta_es,
                                    "evaluacion": evaluacion_social
                                }, f, ensure_ascii=False)
                                # Escribimos una coma y un salto de línea para que se pueda escribir el siguiente elemento
                                f.write(",\n")
                    except Exception as e:
                        print(f"⚠️ Error al evaluar saludo: {e}")


            # Primero comprueba si hay algo en el historial de la conversación.
            # Al haberse inicializado como [], si no hay nada esto es un objeto falsy que será tomado como False
            # En caso de haber algo, comprueba si la función definida en este fichero para buscar la reexplicacion
            # indica que esta es necesaria.
            # En caso de serlo, entramos en a dar la reexplicacion.
            elif self.historial_conversacion:
                necesita_reexplicacion = self.debe_reexplicar_llm(pregunta_traducida)
                if necesita_reexplicacion:
                    # Medimos cuánto tiempo tarda en dar una respuesta reexplicada y su evaluación
                    with cronometro.medir("reexplicacion"):
                        print("🔄 Se detectó que el usuario no comprendió, entrando en módulo de reexplicación.")
                        # Vamos a reescribir el último mensaje del modelo por lo que tenemos que seleccionarlo
                        # Además, lo que vamos a reescribir es la respuesta por lo que tendremos que elegir esa clave
                        # de entre las que guardamos
                        ultimo_mensaje = self.historial_conversacion[-1]["respuesta"]
                        # Generamos la reexplicación a partir del último mensaje del sistema usando la función que 
                        # se encuentra en modulo_reexplicacion.py
                        respuesta_es = modulo_reexplicacion(ultimo_mensaje)
                        # Ahora pasamos a la evaluación de este módulo.
                        # Nota: Esta solo se hace cuando el modelo da una respuesta, es decir, cuando hay historial
                        # previo.
                        try:
                            # Medimos cuanto tiempo tarda en generar la evaluacion
                            with cronometro.medir("evaluacion_reexplicacion"):
                                # Esta función se encuentra en evaluador_generativo.py
                                evaluacion_reexplicacion = evaluar_reexplicacion(ultimo_mensaje, respuesta_es)
                                print("\n🔍 Evaluación de reexplicación:")
                                print(json.dumps(evaluacion_reexplicacion, indent=2, ensure_ascii=False))

                                # Guardamos el resultado de las evaluaciones en un JSON
                                os.makedirs("evaluacion/json", exist_ok=True)
                                with open("evaluacion/json/reexplicaciones.json", "a", encoding="utf-8") as f:
                                    json.dump({
                                        "pregunta": pregunta_traducida,
                                        "original": ultimo_mensaje,
                                        "reexplicacion": respuesta_es,
                                        "evaluacion": evaluacion_reexplicacion
                                    }, f, ensure_ascii=False)
                                    f.write(",\n")
                        except Exception as e:
                            print(f"⚠️ Error al evaluar reexplicación: {e}")
            # En caso de que no haya historial entramos aquí
            # Entonces carga el modelo que comprueba la reexplicacion y si es necesaria da una respuesta por defecto
            # Téngase en cuenta, que tal y como lo hemos hecho se entra en el modelo que lo comprueba una sola vez
            else:
                necesita_reexplicacion = self.debe_reexplicar_llm(pregunta_traducida)
                if necesita_reexplicacion:
                    print("🔄 Se detectó que el usuario no comprendió, entrando en módulo de reexplicación.")
                    respuesta_es = "Disculpe, pero no hay ningún mensaje previo que pueda explicarle. ¿Puedo ayudarle en algo más?"

        # Terminamos este bloque with.
        # Como no hemos puesto ningún return no se sale de la función.
        # Sin embargo, si ha entrado en alguno de los anteriores entonces respuesta_es tiene un valor y no
        # entra dentro del siguiente if si no que pasa directamente a lo siguiente que es la traducción.
        if respuesta_es is None:
            # Si no hay respuesta, iniciamos el crono para ver cuánto tiempo tardamos en resumir el historial
            with cronometro.medir("resumen_historial"):
                # Si hay historial de preguntas del usuario entramos a resumirlo
                if self.historial_preguntas_usuario:
                    # En caso de haber historial y no haber resumen, entramos en la función que se encuentra en resumidor.py

                    # Preparamos la lista de tuplas (pregunta, respuesta)
                    lista_preg_resp = [(item["pregunta"], item["respuesta"]) for item in self.historial_conversacion]

                    # Llamamos a la función obtener_contexto_previo del resumidor.py
                    # Para poder llamarla, hemos tenido que unir el contexto en respuesta y pregunta en una tupla
                    # la función recibe como parámetro un único texto.
                    # a su vez llama a la función para obtener un resumen y nos devuelve el resumen del historial
                    historial_resumido = self.resumidor.obtener_contexto_previo(lista_preg_resp, self.historial_resumen)
                    # Nos guardamos nuestro nuevo historial
                    self.historial_resumen = historial_resumido

                else:
                    historial_resumido = ""
                    print("📄 No hay historial previo relevante, se evaluará la pregunta de forma aislada.")

            # Terminamos el with anterior
            # Ahora vamos a determinar si una pregunta es o no relevante para el RAG.
            with cronometro.medir("evaluacion_relevancia"):
                # Para hacer la comprobación usamos una de las funciones definidas en este archivo.
                relevante = self.es_pregunta_relevante_llm(pregunta_traducida, historial_resumido)

            # En caso de ser relevante para RAG entramos en el siguiente módulo
            if relevante:
                print("✅ La pregunta es relevante para RAG según el historial.")
                # Añadimos al historial de preguntas, la realizada por el usuario. 
                # Esto es porque de cara al historial solo vamos a tener en cuenta las preguntas que entren
                # en el RAG, el resto no se tendrá en cuenta.
                self.historial_preguntas_usuario.append(pregunta_traducida)

                # Inicamos el cronometro para ver cuánto tiempo tarda en el módulo de RAG
                with cronometro.medir("chat_rag_buscar_respuesta"):
                    # Llamamos a la función buscar respuesta que se encuentra dentro de la instancia creada en
                    # chatDoctorado.py
                    respuesta_es, chunks_rankeados = self.chat_rag.buscar_respuesta(
                        pregunta_traducida,
                        self.vectorstore,
                        top_k=self.top_k,
                        contexto_previo=historial_resumido
                    )

                # Si el modelo generativo devuelve la respuesta de que no tiene información, entramos en el if
                if "Lo siento, no tengo información" in respuesta_es:
                    # Indicamos que espera una aclaracion
                    self.esperando_aclaracion = True
                    # Guardamos la última pregunta realizada
                    self.ultima_pregunta_original = pregunta_traducida

                    # Iniciamos el contador de tiempo
                    with cronometro.medir("generar_pregunta_clarificacion"):
                        # Nos vamos ahora a la funcion que se encuentra en clarificador.py
                        # Esto nos da un texto, en concreto una pregunta.
                        respuesta_es = generar_pregunta_clarificacion(pregunta_traducida)
                        # Iniciamos el cronometro para ver cuanto tarda en realizar la evaluacion
                        with cronometro.medir("evaluacion_clarificacion"):
                            try:
                                # Realizamos la evaluación de este módulo usando la función en evaluador_generativo.py
                                # Nos devolverá un JSON aunque se produzcan fallos.
                                # Le damos la pregunta inicial del usuario, y la pregunta para realizar la expansión
                                # que hemos generado
                                evaluacion_clarificacion = evaluar_pregunta_clarificacion(pregunta_traducida, respuesta_es)
                                print("\n🧩 Evaluación de pregunta clarificadora:")
                                # Lo convertimos a objeto JSON
                                print(json.dumps(evaluacion_clarificacion, indent=2, ensure_ascii=False))

                                # Si no existe, creamos la carpeta
                                os.makedirs("evaluacion/json", exist_ok=True)
                                # Abrimos el archivo y guardamos la pregunta inicial, la respuesta generada por 
                                # el modelo y la evaluacion (la cual será un json)
                                with open("evaluacion/json/clarificaciones.json", "a", encoding="utf-8") as f:
                                    json.dump({
                                        "pregunta": pregunta_traducida,
                                        "clarificacion": respuesta_es,
                                        "evaluacion": evaluacion_clarificacion
                                    }, f, ensure_ascii=False)
                                    # Escribimos al final un salto de línea.
                                    f.write(",\n")
                            except Exception as e:
                                print(f"⚠️ Error al evaluar pregunta clarificadora: {e}")

                    print("❓ Se requiere aclaración del usuario.")
                # En caso de que lo que devuelva el modelo sea distinto, lo añadimos al historial como pregunta 
                # y respuesta
                else:
                    self.historial_conversacion.append({
                        "pregunta": pregunta_traducida,
                        "respuesta": respuesta_es
                    })
                    # Iniciamos el crono para medir el funcionamiento del rag
                    with cronometro.medir("evaluacion_rag"):
                        try:
                            # Obtener los chunks relevantes mediante el uso de la funcion en evaluador_rag.py
                            # Esto devuelve todos los chunks cuyo valor de similitud con la pregunta supera el dado
                            chunks_relevantes = obtener_todos_los_chunks_relevantes(
                                pregunta_traducida,
                                self.vectorstore,
                                umbral_similitud=0.75
                            )

                            # En este contexto, chunks_rag se refiere a los top_k documentos seleccionados, con su
                            # url y su texto
                            chunks_rag = chunks_relevantes[:self.top_k]

                            # Extraemos los chunks obtenidos en el reranking, pero solo seleccionamos la url y 
                            # el texto, la puntuación aquí nos da igual.
                            chunks_reranking = [(url, ch) for url, ch, sc in chunks_rankeados]

                            # Le pasamos todos a la función dentro de evaluador_rag.py
                            evaluacion = evaluar_respuesta_con_llm(
                                pregunta_traducida,
                                respuesta_es,
                                chunks_rag,
                                chunks_reranking,
                                chunks_relevantes
                            )
                            # Esto nos devuelve un JSON
                            # Nos aseguramos que chunk_omitidos exista y si no esta ponemos una lista vacía
                            evaluacion["chunks_omitidos"] = evaluacion.get("chunks_omitidos", [])

                            # Mostramos por pantalla las evaluaciones
                            # Si no hay, devolvemos un diccionario vacio.
                            ev = evaluacion.get("evaluacion", {})
                            print("\n📊 Evaluación RAG:")
                            print(f"   - Cobertura: {ev.get('cobertura')}")
                            print(f"   - Precisión: {ev.get('precisión')}")
                            print(f"   - Alucinación: {ev.get('alucinacion')}")
                            print(f"   - Comentario: {ev.get('comentario')}\n")

                            # Creamos las carpetas si no exisetn
                            os.makedirs("evaluacion/json", exist_ok=True)
                            # Ruta del fichero donde vamos a guardar el json
                            ruta_eval_rag = "evaluacion/json/rag.json"

                            # Si la ruta del fichero existe, cargamos los datos con formato JSON
                            if os.path.exists(ruta_eval_rag):
                                with open(ruta_eval_rag, "r", encoding="utf-8") as f:
                                    datos_rag = json.load(f)
                            # En caso contrario, creamos una lista vacia
                            else:
                                datos_rag = []

                            # Añadimos los nuevos datos de evaluacion
                            datos_rag.append(evaluacion)

                            # Guardamos todo en el fichero
                            with open(ruta_eval_rag, "w", encoding="utf-8") as f:
                                json.dump(datos_rag, f, ensure_ascii=False, indent=2)

                            print("✅ Evaluación RAG guardada en evaluacion/json/rag.json")

                        except Exception as e:
                            print(f"⚠️ Error al evaluar RAG: {e}")

            # En caso de no ser considerado relevante para RAG
            else:
                print("❌ Pregunta no relevante para RAG, usando modelo sin contexto.")
                # Iniciamos el cronometro para medir el tiempo que se tarda en el no rag
                with cronometro.medir("responder_no_rag"):
                    # Llamamos a la funcion en modelo_no_rag.py
                    respuesta_es = responder_no_rag(pregunta_traducida)
                    # Lo anterior nos devuelve siempre un texto.

                # Iniciamos el cronómetro para medir el tiempo que se tarda en evaluar esta parte de respuesta
                # general.
                with cronometro.medir("evaluacion_no_rag"):
                    try:
                        # Realizamos la evaluacion de la respuesta general a partir de la función que podemos encontrar
                        # en evaluador_generativo.py
                        evaluacion_no_rag = evaluar_respuesta_no_rag(pregunta_traducida, respuesta_es)
                        print("\n📋 Evaluación modelo sin RAG:")
                        # Esto nso devuelve un JSON con la evaluación del modelo
                        print(json.dumps(evaluacion_no_rag, indent=2, ensure_ascii=False))

                        # Creamos el directorio correspondiente para guardar el archivo con el JSON obtenido
                        # de la evaluación de la pregunta general
                        os.makedirs("evaluacion/json", exist_ok=True)
                        with open("evaluacion/json/no_rag.json", "a", encoding="utf-8") as f:
                            json.dump({
                                "pregunta": pregunta_traducida,
                                "respuesta": respuesta_es,
                                "evaluacion": evaluacion_no_rag
                            }, f, ensure_ascii=False)
                            # Escribimos un salto de línea al final
                            f.write(",\n")
                    except Exception as e:
                        print(f"⚠️ Error al evaluar respuesta sin RAG: {e}")

        # Guardamos unas variables para determinar el idioma al que tenemos que traducir
        # En el caso de ser el idioma original el español entonces no se traduce porque el idioma de destino
        # sera también el español
        # En caso contrario, primero se comprueba si había mezcla de idiomas en cuyo caso el idioma de destino
        # será el inglés
        # En caso de que no hubiese mezcla entonces el idioma al que se traduce es el idioma original en el que 
        # venía el texto.
        if idioma_original == "español":
            idioma_destino = "español"
        elif mezcla:
            idioma_destino = "ingles"
        else:
            idioma_destino = idioma_original

        print(f"🌍 Idioma original: {idioma_original} | Mezcla: {mezcla} | Traducción final: {idioma_destino}")

        # Ponemos un cronometro para medir el tiempo de traduccion de la respuesta final
        with cronometro.medir("traduccion_respuesta_final"):
            if idioma_destino != "español":
                # En caso de no ser el idioma el español, llamamos a la función que se encuentra en traduccion_respuesta.py
                # Esta nos devuelve el texto traducido al idioma correspondiente
                respuesta_final = traducir_respuesta(respuesta_es, idioma_destino)
                print(f"✅ Respuesta traducida del español a: {idioma_destino}")
            else:
                # en caso contrario, la respuesta final será la respuesta original ya que no habría que traducirla
                respuesta_final = respuesta_es
                print("✅ La respuesta está en español, no se traduce.")

        # Iniciamos la evaluación para medir como de bien hace la traducción
        with cronometro.medir("evaluacion_traduccion"):
            # Solo en aquellos casos en los que el idioma no fuera español es cuando tiene sentido ver como 
            # de bien o de mal hace la traducción.
            if idioma_original != "español":
                try:
                    # Llamamos a la funcion en evaluador_traduccion.py, que nos devolverá un JSON
                    resultado_eval = evaluar_traduccion(
                        texto=pregunta_usuario,
                        respuesta=respuesta_es,
                        src_lang=idioma_original,
                        tgt_lang="español"
                    )

                    # Creamos la ruta del fichero si no existe
                    os.makedirs("evaluacion/json", exist_ok=True)
                    ruta_archivo = "evaluacion/json/traduccion.json"

                    # En caso de que el fichero exista, cargamos su contenido
                    if os.path.exists(ruta_archivo):
                        with open(ruta_archivo, "r", encoding="utf-8") as f:
                            datos = json.load(f)
                    else:
                        # En caso contrario creamos una lista vacia
                        datos = []

                    # Añadimos los resultados al fichero
                    datos.append(resultado_eval)

                    # Guardamos el fichero de datos
                    with open(ruta_archivo, "w", encoding="utf-8") as f:
                        json.dump(datos, f, ensure_ascii=False, indent=2)

                    print("✅ Evaluación de traducción guardada en evaluacion/json/traduccion.json")

                except Exception as e:
                    print(f"❌ Error al evaluar o guardar traducción: {e}")

        cronometro.guardar_json()

        return {
            "respuesta": respuesta_final,
            "necesita_aclaracion": self.esperando_aclaracion
        }

    def reiniciar_contexto(self):
        """
        Esta función sirve para reiniciar la conversación.
        
        """
        # Eliminamos los historiales de conversación
        self.historial_conversacion.clear()
        self.historial_preguntas_usuario.clear()
        # Ponemos los valores predefinidos
        self.esperando_aclaracion = False
        self.ultima_pregunta_original = None
        print("🔄 Contexto reiniciado. Puedes empezar una nueva consulta.")

    def chat_interactivo(self):
        """
        Llamamos al chat con el modelo.
        
        """
        # Mensaje inicial
        print(
            "💬 Chat con AsistenteUS (escribe 'salir', 'exit', 'quit' para salir o 'reiniciar' para empezar de nuevo): \n"
            "Soy el asistente virtual de la Universidad de Sevilla sobre temas de doctorados. \n ¿En qué puedo ayudarte?"
        )

        # Mientras que el usuario no se salga
        while True:
            # Recogemos el input del usuario
            user_input = input("\n👤 Tú: ").strip()
            # Si dice de salir entonces reiniciamos el contexto y salimos de la conversación, para dejarlo
            # limpio para la próxima.
            if user_input.lower() in ["salir", "exit", "quit"]:
                self.reiniciar_contexto()
                print("👋 Saliendo del chat...")
                break
            # Si el usuario indica que lo quiere reiniciar, llamamos a la función para reiniciar el contexto
            if user_input.lower() == "reiniciar":
                self.reiniciar_contexto()
                continue
            # Cualquier otra cosa, llamamos a la función responder definida anteriormente
            resultado = self.responder(user_input)
            # Devolvemos la respuesta guardada en el json de resultado al usuario
            print(f"\n🧠 AsistenteUS: {resultado['respuesta']}")


if __name__ == "__main__":
    agente = AgenteDecisor(top_k=5)
    agente.chat_interactivo()