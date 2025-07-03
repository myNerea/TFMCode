import time
import json
import os
from contextlib import contextmanager

class Cronometro:
    def __init__(self):
        self.registros = [] # Creamos un array vacio para ir añadiendo los tiempos por módulo
        self.inicio_total = time.time() # Iniciamos el tiempo en el momento en el que instanciamos la clase
        self.pregunta = None  # Aquí guardamos la pregunta del usuario

    # Función para guardar la pregunta realizada por el usuario
    def set_pregunta(self, pregunta):
        """
        Guardamos la pregunta del usuario.
        
        """
        self.pregunta = pregunta

    # Este decorador permite ejecutar un bloque de código dentro de un with asegurandonos que antes y 
    # después de ello pase algo (ej: cerrar/abrir archivos)
    # Este decorador es un manejador de contexto, todo lo que este después de un with estará dentro del contexto 
    @contextmanager
    def medir(self, nombre_modulo):
        """
        Mide el tiempo que se tarda en cada uno de los módulos.
        """
        # Guardamos el tiempo actual
        inicio = time.time()
        # Cedemos el control para que se ejecute la instrucción dentro del with(mientras tanto el código esta
        # dentro del context manager creado por el decorador @contextmanager)
        yield
        # Guarda el tiempo al terminar el bloque
        fin = time.time()

        # Añadimos un nuevo registro. Este contiene el nombre del módulo(que se ha pasado como parámetro) y 
        # la duración en segundos que duró dentro del módulo
        self.registros.append({
            "modulo": nombre_modulo,
            "duracion_segundos": round(fin - inicio, 4)
        })


    def finalizar(self):
        """
        Esta función se usa para calcula el tiempo total dentro de la función responder.
        Esto devuelve un JSON con el tiempo total, los módulo en los que se ha entrado y sus tiempos y la 
        pregunta que el usuario ha realizado.
        """

        # Calculamos el tiempo total transcurrido (este va desde el llamamiento a la clase hasta el momento
        # presente en el que se llama a la función)
        tiempo_total = round(time.time() - self.inicio_total, 4)

        # Guardamos en un JSON el tiempo total en segundos y el tiempo que se ha tardado en cada uno de los módulos
        # Vemos como esto es un JSON que a su vez contiene varios JSON
        resultado = {
            "tiempo_total": tiempo_total,
            "modulos": self.registros
        }
        # Realizamos por si acaso la comprobación sobre la pregunta(aunque se espara que siempre exista)
        if self.pregunta is not None:
            # En el caso de existir añadimos al final del JSON una par clave/valor con pregunta:preguntaUsuario
            resultado["pregunta"] = self.pregunta

        # Devolvemos el JSON
        return resultado

    def guardar_json(self, nombre_archivo="evaluacion/json/tiempos.json"):
        """
        Guardamos los tiempos en un fichero JSON.
        Nota: El fichero no tendrá un formato JSON perfecto, ya que contendra una lista de diccionarios de la 
        siguiente forma {},{} y para que fuese correcto tendría que ser [{},{}]. Esto es necesario tenerlo en 
        cuenta de cara a un posterior tratamiento de los datos.

        """
        # Si la carpeta evaluacion o json o ambas no existen, se crean.
        os.makedirs("evaluacion/json", exist_ok=True)
        # Llamamos a la función definida previamente
        datos = self.finalizar()
        # Abrimos el archivo en modo append(por eso lo de "a". El modo append es para añadir al final sin borrar lo anterior)
        with open(nombre_archivo, "a", encoding="utf-8") as f:
            # Codifica como JSON el diccionario creado arriba
            # ensure_ascii=False asegura que los caracteres especiales se guarden correctamente (no los convierte en secuencias \uXXXX).
            json.dump(datos, f, ensure_ascii=False)
            # Después de añadir el JSON, añade una coma
            f.write(",\n")
