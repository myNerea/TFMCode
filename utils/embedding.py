import os
import json
import aiohttp
import asyncio
import requests
import hashlib
import io
import re
import numpy as np
from urllib.parse import urljoin, urlparse
from lxml import etree
from PyPDF2 import PdfReader
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer

# Guardamos la ruta absoluta de este archivo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# A partir de la ruta anterior hacemos la búsqueda de la carpeta index
INDEX_DIR = os.path.join(BASE_DIR, "index")
# A partir de la ruta anterior hacemos la búsqueda de el archivo vectorstore
VECTORSTORE_PATH = os.path.join(INDEX_DIR, "vectorstore.json")
# Guardamos el número máximo de tokens
MAX_TOKEN = 512
# Guardamos el overlap que vamos a usar.
OVERLAP = 50
# Si no existe creamos la carpeta index
os.makedirs(INDEX_DIR, exist_ok=True)

# Hacemos una lista para excluir los patrones que requieran autentificación
EXCLUDE_PATTERNS = [
    r"login", r"signin", r"admin", r"auth", r"usuario", r"user",
    r"account", r"session", r"perfil", r"register", r"signup"
]

def url_excluida(url):
    """
    Recibe una url y determina si esta coincide con alguno de los patrones a excluir.
    Devuelve True o False en función de si coincide algún patrón con los que se quiere excluir o no.

    """
    # Comprueba uno a uno los elementos de la lista excluir para ver si el patron coincide con el de la url
    # Si coincide en al menos uno devuelve True, en caso contrario devuelve False
    return any(re.search(pat, url, re.IGNORECASE) for pat in EXCLUDE_PATTERNS)

def leer_pdf_desde_url(url):
    """
    Esta función recibe una url de un pdf y devuelve el texto que contiene el pdf.
    
    """

    # Descargamos el contenido de la url
    response = requests.get(url)
    # Si la respuesta no fue exitosa, es decir, la descarga falló, lanza un error 
    if response.status_code != 200:
        raise Exception(f"❌ No se pudo descargar el PDF desde {url}")
    # En caso de que la respuesta fuese exitosa, crea un objeto de bytes en memoria a partir del PDF, el cual
    # se cierra automaticamente cuando termina el with
    with io.BytesIO(response.content) as f:
        # Creamos un lector de pdf a partir del contenido en memoria para poder leer el contenido y las páginas
        reader = PdfReader(f)
        # Inicializamos una lista vacía para guardar el texto de las páginas
        texto_paginas = []
        # Iteramos sobre todas las páginas del PDF, siendo i el indice de la página y page el objeto que la representa
        for i, page in enumerate(reader.pages):
            # Extraemos el texto de cada página
            texto = page.extract_text()
            # Si la página tenía texto, no todas las páginas tenían porque tenerlo, podían tener imágenes.
            if texto:
                # En caso de haber texto se añade a la lista anterior indicando el número de la página.
                texto_paginas.append(f"\n--- Página {i+1} ---\n{texto}")
    # Lo devolvemos uniendo cada uno de los elementos de la lista y separándolos con un salto de línea.
    return "\n".join(texto_paginas)



def leer_html_o_xml_desde_url(url):
    """
    Devolvemos el texto limpio que se encuentra dentro de las url que tienen HTML o XML.
    
    """

    # Igual que antes descargamos el contenido de la url y si no fue éxitosa lanzamos un error.
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"❌ No se pudo descargar la página desde {url}")

    # En caso de ser éxitosa, primero detectamos el tipo de encabezado que tiene, para ver si es xml o html.
    # Si no hay encabezado nos devuelve la cadena vacía.
    # En caso de haberlo, convierte el valor que devuelva a minusculas.
    content_type = response.headers.get("Content-Type", "").lower()

    # Comprobamos si o bien la cabecera de la url contiene la palabra xml o bien la url termina en xml
    if "xml" in content_type or url.lower().endswith(".xml"):
        # En caso de ser xml crea un parser (analizador) concreto para este elemento
        # El recover=True es para que intente recuperarse de errores. Esto es por si encuentra un xml mal formado
        parser = etree.XMLParser(recover=True)
    else:
        # En caso contrario creamos un parser para html, el cual también intenta recomponerse de errores por malformaciones en el fichero
        parser = etree.HTMLParser(recover=True)

    # Tomamos el contenido de la respuesta, que es un html o un xml y lo convierte a un árbol de elementos que 
    # representa el documento. 
    # Esto lo hace usando el parser creado anteriormente.
    tree = etree.fromstring(response.content, parser)

    # Con el siguiente bucle eliminamos los estilos
    # Buscamos los elementos en el árbol que sean de tipo script, style y noscript, mediante xpath.
    # Esto obtiene una lista de nodos
    for tag in tree.xpath("//script | //style | //noscript"):
        # para cada nodo encontrado, busca su elemento padre
        parent = tag.getparent()
        # Si el nodo padre existe, se elimina el hijo.
        if parent is not None:
            parent.remove(tag)

    # A continuación, obtenemos una lista de todos los nodos del texto con tree.xpath, es decir, todo el texto dentro del documento
    # Para cada texto encontrado se eliminan los espacios delante y detrás.
    # Aquellos que no se queden vacíos tras hacer esto son los que se incluyen en la lista
    # Después unimos todos los elementos de la lista en un único texto separándolos con saltos de líneas.
    texto = "\n".join([t.strip() for t in tree.xpath("//text()") if t.strip()])
    # Devolvemos el texto limpio.
    return texto

def extraer_pdfs_y_urls(url_base):
    """
    Esta función se encarga de extraer el contenido HTML de una web dada para extraer:
        - Los enlaces a archivos PDF que se encuentren dentro de este sitio web.
        - Otros enlaces dentro del mismo sitio que no sean PDFs.

    """

    # Igual que antes descargamos el contenido de la url y si no fue éxitosa lanzamos un error.
    response = requests.get(url_base)
    if response.status_code != 200:
        raise Exception(f"❌ No se pudo descargar la página desde {url_base}")

    # Creamos un parser especifico para HTML, el cual le pedimos que se recupere de errores por malformaciones.
    parser = etree.HTMLParser(recover=True)
    # Creamos el árbol de elementos que representa el documento
    tree = etree.fromstring(response.content, parser)

    # Eliminamos los elementos que sean script, style y noscript
    for tag in tree.xpath("//script | //style | //noscript"):
        parent = tag.getparent()
        if parent is not None:
            parent.remove(tag)

    # Creamos dos conjuntos:
    # El primero para guardar URL únicas (que no se repitan)
    links = set()
    # El segundo para almacenar PDFs sin repetirlos.
    pdfs = set()

    # Buscamos todos los elemento a que tengan un href dentro del árbol. Buscamos esto porque es donde se
    # especifica la url o dirección a la que apunta el enlace.
    for a in tree.xpath("//a[@href]"):
        # Obtenemos el valor del atributo href.
        href = a.attrib['href']
        # Usamos la url base para unirla con el enlace y tener asi una url entera funcional
        full_url = urljoin(url_base, href)

       # Si la url esta dentro de las excluidas, pasa directamente al siguiente enlace saltandose el resto
       #  del código 
        if url_excluida(full_url):
            continue

        # lo siguiente analiza y descompone cada una de las url en sus componentes (protocolo,dominio,parámetros, etc)
        parsed_base = urlparse(url_base)
        parsed_full = urlparse(full_url)

        # Si el dominio y el puerto son el mismo para la ruta base y para la nueva y además, la nueva ruta
        # empieza por la ruta base, entonces entramos dentro.
        # Esto se hace para verificar que estamos dentro del mismo dominio. Lo hacemos porque no queremos salirnos
        # de las url de doctorados. 
        if (parsed_full.netloc == parsed_base.netloc and
            full_url.startswith(url_base)):
            # comprobamos si la url termina con pdf en cuyo caso lo añadimos al objeto de pdfs
            if full_url.lower().endswith(".pdf"):
                pdfs.add(full_url)
            # en caso contrario lo añadimos al objeto de links
            else:
                links.add(full_url)

    # convertirmos los set en listas y devolvemos una tupla de dos elementos.
    return list(pdfs), list(links)


# Carga tokenizer compatible con LLama, que es parecido a llama3.2:3b
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")

def contar_tokens(texto):
    # Recibimos un texto, usamos el tokenizador definido anteriormente para dividirlo en tokens
    tokens = tokenizer.tokenize(texto)
    # Devolvemos el número de tokens.
    return len(tokens)

def dividir_en_chunks(texto, max_tokens=MAX_TOKEN, token_overlap=OVERLAP, idioma='spanish'):
    """
    Dividimos un texto largo en unos fragmentos más pequeños (chunks) que tendrán un límite máximo de tokens.
    Para esta división, intentamos respetar las frases completas. Además, usamos un overlap para no perder contexto
    de los fragmentos.
    
    """
    # Dividimos el texto en frases en base al idioma (español)
    frases = sent_tokenize(texto, language=idioma)
    # Creamos las listas para guardar posteriormente los chunks
    chunks = []
    buffer = []
    buffer_tokens = 0

    # Para cada una de las frases, contamos el número de tokens que tiene.
    for frase in frases:
        frase_tokens = contar_tokens(frase)

        # Si la frase es muy grande, tiene más tokens que el número máximo de palabras,
        # se puede fragmentar por palabras o tokens
        if frase_tokens > max_tokens:
            # Dividimos la frase en palabras
            palabras = frase.split()
            sub_buffer = []
            sub_tokens = 0
            
            for palabra in palabras:
                # Para cada palabra contamos el número de tokens
                palabra_tokens = contar_tokens(palabra)
                # Vemos si al agregar la palabra superamos el límite
                if sub_tokens + palabra_tokens > max_tokens:
                    # Si se supera, guarda lo acumulado hasta ahora como chunk
                    chunks.append(" ".join(sub_buffer))

                    # Ahora meteremos esa palabra en un nuevo chunk junto con parte del overlap del chunk anterior
                    overlap = []
                    overlap_tokens = 0
                    # Tomamos las últimas palabras del chunk anterior
                    for w in reversed(sub_buffer):
                        # Contamos el número de token de cada palabra
                        t = contar_tokens(w)
                        # Si al añadir esa palabra a los tokens que hay superamos el Overlap entonces salimos
                        # del bucle
                        if overlap_tokens + t > token_overlap:
                            break
                        # en caso contrario, insertamos la palabra al principio de la lista para mantener el orden
                        # original
                        overlap.insert(0, w)
                        # recalculamos el número de tokens que llevamos para el overlap
                        overlap_tokens += t

                    # Metemos la palabra dentro del sub_buffer con el overlap correspondiente
                    sub_buffer = overlap + [palabra]
                    # Recalculamos el numero de tokens
                    sub_tokens = overlap_tokens + palabra_tokens
                else:
                    # En caso contrario lo añade al buffer y aumenta el recuento de tokens
                    sub_buffer.append(palabra)
                    sub_tokens += palabra_tokens
            # Si queda algo por guardar después de procesar las frases, entonces lo guarda en un nuevo chunk
            if sub_buffer:
                chunks.append(" ".join(sub_buffer))
            # Tras llegar aquí usamos el continue para que siga con la siguiente frase, y ignore el código de abajo
            continue

        # En el caso en el que la frase por si misma no supere el número máximo de tokens, se añade al buffer
        # en caso de que al agregarla a lo que ya hay, se supera el tamaño del chunk, entonces se crea uno nuevo.
        if buffer_tokens + frase_tokens > max_tokens:
            # Guardamos lo que llevemos de frases (buffer) en un chunk
            chunks.append(" ".join(buffer))
            # Creamos los elementos necesarios para recoger el solapamiento
            overlap = []
            overlap_tokens = 0
            # Recorremos al reves el buffer, es decir, empezamos por las últimas frases que se hayan añadido
            for frase_solap in reversed(buffer):
                # Contamos el número de tokens de las frases
                t = contar_tokens(frase_solap)
                # Si este número en conjunto con los que ya haya de overlap supera el máximo, entonces dividimos
                # la frase en palabras para intentar rellenarlo con ellas.
                if overlap_tokens + t > token_overlap:
                    # Dividimos la frase en palabras
                    palabras = frase_solap.split()
                    # Creamos los elementos necesarios para recoger las palabras
                    sub_overlap = []
                    sub_tokens = 0
                    # Empezamos por las últimas palabras de la frase en el bucle
                    for palabra in reversed(palabras):
                        # Para cada palabra contamos el número de tokens
                        palabra_tokens = contar_tokens(palabra)
                        # Si el número de tokens de la palabra unido a los que ya hay supera al máximo de overlap,
                        # salimos
                        if sub_tokens + palabra_tokens > token_overlap:
                            break
                        # en caso contrario insertamos la palabra al principio y añadimos el nuevo número de tokens
                        sub_overlap.insert(0, palabra)
                        sub_tokens += palabra_tokens
                    # si tiene algo, se inserta al principio de overlap y se le hace el recuento del número de tokens
                    if sub_overlap:
                        overlap.insert(0, " ".join(sub_overlap))
                        overlap_tokens += sub_tokens
                    # Y salimos del for
                    break
                # En caso contrario, añadimos la frase al principio para mantener el orden y aumentamos el recuento
                # de tokens
                else:
                    overlap.insert(0, frase_solap)
                    overlap_tokens += t
            
            # Creamos una copia de overlap en buffer para tener recogidos ya los nuevos elementos
            buffer = overlap.copy()
            # Reescribimos el recuento de tokens
            buffer_tokens = overlap_tokens

        # Añadimos la frase al buffer en caso contrario
        buffer.append(frase)
        # Añadimos el recuento del número de tokens
        buffer_tokens += frase_tokens

    # Si se queda algo cuando terminemos dentro de buffer lo metemos en un nuevo chunk
    if buffer:
        chunks.append(" ".join(buffer))

    # Devolvemos los chunks
    return chunks


async def obtener_embedding_ollama(texto, modelo="mxbai-embed-large"):
    """
    Función para obtener la representación vectorial de las frases.
    
    """

    url = "http://localhost:11434/api/embeddings"
    data = {"model": modelo, "prompt": texto}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as response:
            if response.status == 200:
                # En el caso de que la petición haya ido bien, devolvemos los embedding obtenidos
                respuesta_json = await response.json()
                return respuesta_json["embedding"]
            else:
                # Si ha ido mal devolvemos un error.
                texto_error = await response.text()
                raise Exception(f"❌ Error al generar embedding: {texto_error}")

def cargar_vectorstore():
    """
    Función para cargar el contenido de la base de datos vectorial
    
    """
    # Si no existe la ruta del archivo, entonces devuelve un diccionario vacio
    if not os.path.exists(VECTORSTORE_PATH):
        return {}
    # Si existe, la abre, lo lee y devuelve su contenido como un diccionario.
    with open(VECTORSTORE_PATH, "r") as f:
        return json.load(f)

def guardar_vectorstore(vectorstore):
    """
    Función para escribir en la base de datos vectorial.

    """
    # Guardamos el contenido en formato JSON
    with open(VECTORSTORE_PATH, "w") as f:
        json.dump(vectorstore, f, indent=2)

async def procesar_una_pagina_y_recursos(url_principal):
    # Cargamos la base de datos vectorial existente (si es que existe)
    vectorstore = cargar_vectorstore()
    # Creamos un diccionario para guardar los datos actualizados
    nuevos_vectorstore = {}

    async def procesar_url(url, tipo):
        """
        Función para realizar el procesamiento de la url en función de su tipo

        """
        try:
            # Generamos un hash (identificador) único de la URL
            hash_url = hashlib.sha256(url.encode()).hexdigest()
            # Si la URL ya fue procesada antes y no ha cambiado, se reutiliza lo guardado y se omite el reprocesamiento.
            if url in vectorstore and vectorstore[url]["hash"] == hash_url:
                print(f"✅ {tipo} sin cambios: {url}")
                nuevos_vectorstore[url] = vectorstore[url]
                return
            # En caso contrario lo procesamos en función del tipo
            print(f"🔄 Procesando {tipo}: {url}")
            # Si es PDF llamamos a la función para leer los pdf, si no llamamos a la otra
            if tipo == "PDF":
                texto = leer_pdf_desde_url(url)
            else:
                texto = leer_html_o_xml_desde_url(url)

            # Después dividimos en chunks el texto obtenido
            chunks = dividir_en_chunks(texto)
            # Creamos una lista para guardar los embedding
            chunks_con_embeddings = []
            # Para cada uno de los chunks creamos la representación vectorial y lo guardamos en un diccionario
            # junto al texto que le corresponde (el texto que había en el chunk).
            for chunk in chunks:
                embedding = await obtener_embedding_ollama(chunk)
                chunks_con_embeddings.append({
                    "texto": chunk,
                    "embedding": embedding
                    })

            # Añadimos el nuevo elemento como un diccionario que contiene primero el identificiador hash
            # y despues un diccionario con el texto y el embedding
            nuevos_vectorstore[url] = {
                "hash": hash_url,
                "chunks": chunks_con_embeddings
            }
        except Exception as e:
            print(f"❌ Error con {tipo} {url}: {e}")

    # Como le tenemos puesto Página principal, no entra en lo de PDF y se analiza como una url
    procesar_url(url_principal, "Página principal")

    try:
        # Intentamos extraer las listas con las url que contienen los pdf y los html de la página principal
        # Solo se búsca dentro de la página principal.
        pdf_urls, html_urls = extraer_pdfs_y_urls(url_principal)
    except Exception as e:
        print(f"❌ Error extrayendo PDFs/URLs: {e}")
        pdf_urls, html_urls = [], []

    # para cada una de las páginas que pertenecían a pdf le ponemos el tipo pdf
    for url in pdf_urls:
        procesar_url(url, "PDF")

    # para las que pertenecían a html/xml lo llamamos url
    for url in html_urls:
        procesar_url(url, "URL")

    # Guardamos la base de datos vectorial
    guardar_vectorstore(nuevos_vectorstore)
    # Devolvemos la nueva base de datos vectorial
    return nuevos_vectorstore

if __name__ == "__main__":
    url_inicial = "https://doctorado.us.es/estudios/programas-de-doctorado"
    print("⏳ Procesando página y recursos, por favor espera...")
    # Llamamos a la función anterior pasandole la url principal.
    vectorstore =  asyncio.run(procesar_una_pagina_y_recursos(url_inicial))
    print("✅ Vectorstore cargado y actualizado.")
