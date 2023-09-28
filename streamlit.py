import streamlit as st
import tempfile
import os
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferMemory
import pinecone
from langchain.chains.question_answering import load_qa_chain
from utils.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains import LLMChain
from streamlit_chat import message

openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pinecone_environment = st.secrets["PINECONE_ENVIRONMENT"]
pinecone_index = st.secrets["PINECONE_INDEX_NAME"]
pinecone_namespace = st.secrets["PINECONE_NAMESPACE"]


def ingest_docs(temp_dir: str = tempfile.gettempdir()):
    try:
        if not openai_api_key or not pinecone_api_key or not pinecone_environment or not pinecone_index or not pinecone_namespace:
            raise ValueError(
                "Пожалуйста укажите необходимый набор переменных окружения")
        loader = DirectoryLoader(
            temp_dir, glob="**/*.pdf", loader_cls=PyPDFLoader, recursive=True
        )
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100)
        documents = text_splitter.split_documents(documents)

        pinecone.init(
            api_key=pinecone_api_key, environment=pinecone_environment)

        embeddings = OpenAIEmbeddings(
            model='text-embedding-ada-002', openai_api_key=openai_api_key)

        Pinecone.from_documents(
            documents, embeddings, index_name=pinecone_index, namespace=pinecone_namespace)
    except Exception as e:
        st.error(f"Возникла ошибка при добавлении ваших файлов: {str(e)}")

def main():
    st.title('Чат с вашими PDF файлами')
    st.write('Загружайте свои PDF-файлы и задавайте вопросы по ним. Если вы уже загрузили свои файлы, то переходите к чату ниже.')

    if not openai_api_key or not pinecone_api_key or not pinecone_environment or not pinecone_index or not pinecone_namespace:
        st.warning(
            "Пожалуйста, задайте свои учетные данные в streamlit secrets для запуска этого приложения.")

    uploaded_files = st.file_uploader(
        "После загрузки файлов в формате pdf начнется их инъекция в векторную БД.", accept_multiple_files=True, type=['pdf'])

    if uploaded_files:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
                    file_name = uploaded_file.name
                    with open(os.path.join(temp_dir, file_name), "wb") as f:
                        f.write(uploaded_file.read())
                with st.spinner("Добавление ваших файлов в базу ..."):
                    ingest_docs(temp_dir)
                    st.success("Ваш(и) файл(ы) успешно принят(ы)")
                    st.session_state['ready'] = True
        except Exception as e:
            st.error(
                f"При загрузке ваших файлов произошла ошибка: {str(e)}")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    if 'ready' not in st.session_state:
        st.session_state['ready'] = True

    if st.session_state['ready']:

        pinecone.init(
            api_key=pinecone_api_key, environment=pinecone_environment)

        embeddings = OpenAIEmbeddings(
            model='text-embedding-ada-002', openai_api_key=openai_api_key)

        llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0,
                         openai_api_key=openai_api_key, verbose=True)

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        vectorstore = Pinecone.from_existing_index(
            index_name=pinecone_index, embedding=embeddings, text_key='text', namespace=pinecone_namespace)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        question_generator = LLMChain(
            llm=llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=True)
        doc_chain = load_qa_chain(
            llm, chain_type="stuff", prompt=QA_PROMPT, verbose=True)

        qa = ConversationalRetrievalChain(
            retriever=retriever, question_generator=question_generator, combine_docs_chain=doc_chain, verbose=True, memory=memory)

        if 'generated' not in st.session_state:
            st.session_state['generated'] = [
                "Что бы вы хотели узнать о документе?"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Привет!"]

        response_container = st.container()

        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input(
                    "Вопрос:", placeholder="О чем этот документ?", key='input')
                submit_button = st.form_submit_button(label='Отправить')

            if submit_button and user_input:
                with st.spinner("Думаю..."):
                    print("История чата: ", st.session_state['chat_history'])
                    output = qa(
                        {"question": user_input})
                    print(output)
                    st.session_state['past'].append(user_input)
                    st.session_state['generated'].append(output['answer'])

                    st.session_state['chat_history'].append(
                        {"вопрос": user_input, "ответ": output['answer']})

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(
                        i) + '_user')
                    message(st.session_state["generated"][i], key=str(
                        i))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.write(f"Что-то пошло не так. Пожалуйста, попробуйте еще раз. {str(e)}")
