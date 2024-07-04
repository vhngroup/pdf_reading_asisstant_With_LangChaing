from langchain_community.document_loaders import PyPDFLoader #Importar PDF
from langchain.text_splitter import RecursiveCharacterTextSplitter #Seccionamos los PDFs
from langchain_community.vectorstores import Chroma #Bases de datos locales
import streamlit.components.v1 as components
from langchain_openai import OpenAIEmbeddings, ChatOpenAI #Trabajar con OPENAI
from langchain.chains import LLMChain #Uso de cadenas de lenguaje natural
from tempfile import NamedTemporaryFile
from urllib.parse import unquote
from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder #Uso de Prompt prearmados.
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import os # Manejo de archivos
import shutil # 

Api_Key = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = Api_Key

#def actualizar_embeddings(url_document):
#        name_file = unquote(Path(url_document).name)
#        ruta_pdf = "./data/"+name_file
#        print(ruta_pdf)


#if __name__== "__main__":
#    url_document = input(r"Por favor indique la ruta del documento")
#    actualizar_embeddings(url_document)

extensions = ["pdf", "docx"]
name_file = ""
st.session_state.disabled = True
status_upload = False
def state_button():
    #url_root = url.split("tmp", 1)[0] optenemos la URL final
    #name_file = url.split("\\", -1)[-1] Optenemos el nombre del fichero temporal
    st.session_state.disabled = False
    status_upload = True

def actualizar_embeddings(name_file): #Cargamos el documento
    ruta_pdf = "./data/"+name_file
    st.write("Por favor espere, se esta procesando el archivo.")
    loader=PyPDFLoader(ruta_pdf)
    docs = loader.load()
    #Dividimos el texto, en segmentos de 1000 caracteres, 50 caracterres de contexto anterior.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    #Guardamos en una variable el documento fragmentado.
    chunked_documents = text_splitter.split_documents(docs)

    #creamos una base de datos vectorial
    vectordb = Chroma.from_documents(
        chunked_documents, OpenAIEmbeddings(model="text-embedding-3-large", api_key=Api_Key), 
        persist_directory="./choma_db"    
        )
    f.close()
    os.unlink(f.name)

# Creamos nuestra interfas web

st.title("Mesa de ayuda para leer documentos")

st.write("Este agente responde preguntas asociadas al documento")

uploaded_file = st.file_uploader("Cargar Documento", type=extensions, accept_multiple_files=False, disabled=status_upload)
if uploaded_file is not None:    
    with NamedTemporaryFile(dir=r"./data/", suffix=".pdf", delete=False) as f:
        f.write(uploaded_file.getbuffer())
    name_file = unquote(Path(f.name).name)
    state_button()
    uploaded_file = None
    

if st.button("Actualizar Embeddings", key="embeddings", disabled=st.session_state.disabled):
    vectordb = actualizar_embeddings(name_file)
    st.write("Embeddings actualizados correctamente")

pregunta = st.text_area("Que pregunta deseas realizar sobre el documento")

prompt_template = """
Eres un agente de ayuda inteligente especializado en la lectura y analisis de documentos.
Responde las preguntas de los usuarios {input} relacionadas con el documento entregado y estrictamente en el {context} proporcionado.
No hagas suposiciones ni proporciones información no en este incluida en el {context}
"""
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", max_tokens=1024, api_key=Api_Key)
qa_chain = LLMChain(llm=llm, prompt= PromptTemplate.from_template(prompt_template))

if st.button("Enviar"):
    if pregunta:
        vectordb = Chroma(persist_directory="./choma_db",
            embedding_function = OpenAIEmbeddings(model="text-embedding-3-large", api_key=Api_Key),
               
        )
        #Hacemos la busqueda por similaridad y traemos las 5 mas paecidas
        resultados_similares = vectordb.similarity_search(pregunta, k=5)
        contexto=""

        for doc in resultados_similares:
            contexto += doc.page_content

        respuesta = qa_chain.invoke({"input": pregunta, "context": contexto})
        resultado = respuesta["text"]
        st.write(resultado)

    elif f.name == "" :
        st.write("Por favor ejecute el embedding")
    else:
        st.write("Por favor ingresa una pregunta antes de enviar")