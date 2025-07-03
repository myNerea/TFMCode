# TFM

## Breve descripción
En este proyecto TFM encontraras un sistema RAG que te podrá resolver las dudas sobre doctorados en la Universidad de Sevilla. Este sistema, ha sido creado integramente con datos públicos a través de la propia Universidad, usando como base para la programación Python.

## Modo de uso
Para poder usar el AsistenteUs por tu mismo, sigue los siguiente pasos:

**Paso 1:** Importar librerías:
pip install -r requirements.txt

**Paso 2:** Descargar modelos Ollama:
Modelo Generativo: ollama pull llama3.2
Modelo Embedding: ollama pull mxbai-embed-large

*Nota:* Para este paso, primero tienes que tener descargado Ollama en tu dispositivo. Si no lo tienes aun, sigue las instrucciones dadas en su página web: https://ollama.com/


**Paso 3:** Cargar función principal:
python agente_decisor.py

**Paso 4:** ¡Disfrute de sus consultas!, y trate de no romperlo 😉.