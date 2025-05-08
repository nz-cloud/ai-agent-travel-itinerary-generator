import os
from dotenv import load_dotenv
load_dotenv()

# A CLASSE PARA OS MODELOS DE LLMs DA OPENAI
from langchain_openai import ChatOpenAI

# CHATPROMPTTEMPLATE = CRIA OS TAMPLETES DE PROMPT COMPATIVEIS COM O CHAT. |  MESSAGESPLACEHOLDER = RESERVA UM HISTÓRICO DE MENSAGENS PARA O PROMPT
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#RUNNABLWITHMESSAGEHISTORY = GERENCIADOR DO HISTÓRICO DE CONVERSA
from langchain_core.runnables.history import RunnableWithMessageHistory

#BASECHATMESSAGEHISTORY E CHATMESSAGEHISTORY = ARMAZENA UM HISTÓRICO DA NOSSA CONVERSA EM UM FORMATO ESTRUTURADO
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

template = """Você é um Assistente de Viagem que ajuda o usuário a planejar viagens, dando sugestões de destinos, roteiros e dicas práticas.
A primeira coisa que deve fazer é perguntar para aonde o usuário vai, com quantas pessoas e por quanto tempo.

Histórico de conversa:
{history}

Entrada do usuário:
{input}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    MessagesPlaceholder(variable_name="history"), #Placeholder para o histórico estruturado
    ("human", "{input}")
])

llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")

chain = prompt | llm

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Fazendo encadeamento do nosso historico de conversa
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"

)

# Criar a função principal
def iniciar_assistente_viagem():
    print("Bem-vindo ao Assistente de viagem! Digite 'sair' para encerrar. \n")
    while True:
        pergunta_usuario = input("Você: ")
         
        if pergunta_usuario.lower() in ["sair", "exit"]:
            print("Assistente de Viagem: Até mais! Aproveite sua viagem!")
            break
        
        # Para iniciar o chat
        resposta = chain_with_history.invoke(
            {'input': pergunta_usuario},
            config={'configurable': {'session_id': 'user123'}}
        )

        print(f"Assitente de viagem: {resposta.content}")


if __name__ == "__main__":
    iniciar_assistente_viagem()