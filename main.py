import chainlit as cl
from agente_decisor import AgenteDecisor

AUTHOR_CHAINLIT = "AsistenteUS"

WELCOME_MESSAGE = """
👋 **Hola, soy AsistenteUS**, tu asistente inteligente para doctorados de la Universidad de Sevilla.

Puedo ayudarte con:
- 🎓 Información sobre programas de doctorado
- 📝 Procedimientos administrativos, plazos y requisitos
- 📚 Acceso a documentos y normativas
- 🤝 Soporte para dudas frecuentes

**¿En qué puedo ayudarte hoy?**
"""

agente = AgenteDecisor(top_k=10)

@cl.on_chat_start
async def on_chat_start():
    agente.reiniciar_contexto()
    await cl.Message(content=WELCOME_MESSAGE, author=AUTHOR_CHAINLIT).send()

@cl.on_message
async def on_message(message: cl.Message):  
    texto_usuario = message.content
    
    # Aquí usamos await para la llamada asincrónica
    resultado = await agente.responder(texto_usuario)

    print(f"🧾 Respuesta enviada al usuario:\n{resultado['respuesta']}\n")

    await cl.Message(
        content=resultado['respuesta'],
        author=AUTHOR_CHAINLIT
    ).send()
