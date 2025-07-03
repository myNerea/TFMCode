import json
from agente_decisor import AgenteDecisor

def main():
    with open("preguntas.json", "r", encoding="utf-8") as f:
        preguntas = json.load(f)

    agente = AgenteDecisor(top_k=10)

    for categoria, lista_preguntas in preguntas.items():
        for pregunta in lista_preguntas:
            respuesta = agente.responder(pregunta)
            print(f"[{categoria}] Pregunta: {pregunta}")
            print(f"Respuesta: {respuesta}")
            print("-" * 40)


if __name__ == "__main__":
    main()
