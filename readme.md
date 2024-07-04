# Creamos un asistence virtual al cual le pasas un archivo PDF y con preguntas de lenguaje natural, te respondera de acuerdo al contenido suministrado.
Potenciado con inteligencia artificial, usando LangChain, OpenAI, Chromadb para persistencia y Streamlit para un despligue web rapido.
![Esquema](https://github.com/vhngroup/pdf_reading_asisstant_With_LangChaing/blob/main/img/flow.png)
## Herramienta:
Este asistente es un ejemplo en el cual se le suministra un archivo PDF el cual deseas analizar y luego a través del prompt y de preguntas de lenguaje natural, se le puede pedir al modelo que respondan a interrogantes 
* Se ha usado la API de OpenAI junto a LangChain y el modelo "gpt-3.5-turbo-16k".
* Se ha usado la tecnica de "TextSplitter" en la cual el documento es fraccionado en segmentos mas pequeños para posteriormente crear los Embeddings.
* Se hace uso de la base de datos vectorial Chromadb, con la cual se da persistencia al entrenamiento, con lo cual no es necesario procesar nuevamente el documento, luego de haber sido procesado por primera vez.
## Uso:
* Recomendamos siempre usar entorno virtual venv.
* Instalamos las dependencias: ``` pip install -r requirements.txt ```
* Ejecutamos el script: ``` streamlit run main.py ```
* Se debe contar con una cuenta de OpenAI y crear en ella un proyecto y crear una contraseña de API, usar la variable "OPENAI_API_KEY".
* Damos click en la caja de carga de documentos y seleccionamos el documento PDF a analizar.
* Hacemos click en "cargar embedings" y esperamos al mensaje de analisis terminado.
* Realizamos la pregunta que deseamos que el modelo nos responda y hacemos click en el boton enviar.

## Posibles errores:
* Building wheel for chroma-hnswlib (pyproject.toml) did not run successfully
    * Es por que no se tiene instalado unas librerias de visual studio, en stackoverflow esta paso a paso como solucionarlo, descargando el visual cpp build tools.

## Agradecimientos:
Este proyecto, tomo como base el realizado en el canal de Youtube @CodigoEspinoza, comparto link del video: https://www.youtube.com/watch?v=1TLtt7kjD5s