from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import chromadb
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import fitz  # PyMuPDF para leitura de PDF
import os
from datetime import datetime
import prompts


# Configurar API da OpenAI
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


def extrair_texto_pdf(caminho_pdf):
    
    # Extrai o texto de um arquivo PDF.
    doc = fitz.open(caminho_pdf)
    texto_completo = ""
    
    for pagina in doc:
        texto_completo += pagina.get_text("text") + "\n"
    
    return texto_completo


def criar_ou_carregar_base_vetorial(texto):
    # Usa ChromaDB para armazenar vetores de forma persistente.
    
    # Criar banco de dados Chroma persistente
    db = chromadb.PersistentClient(path="chroma_db")

    # Nome da coleção no ChromaDB
    collection_name = "langchain"

    # Verificar se a coleção já existe
    existing_collections = db.list_collections()  # Retorna nomes das coleções
    collection_names = [col for col in existing_collections]  
    if collection_name in collection_names:  

        print("🔄 ##### Banco vetorial encontrado! Carregando dados...")
        vetorstore = Chroma(persist_directory="chroma_db", embedding_function=OpenAIEmbeddings())
    else:
        print(" ##### Criando novo banco vetorial...")
        
        # Divide o texto em pedaços menores para melhor indexação
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        textos_divididos = splitter.split_text(texto)

        # Criar base vetorial com Chroma
        vetorstore = Chroma.from_texts(textos_divididos, OpenAIEmbeddings(), persist_directory="chroma_db")
        # vetorstore.persist()  # Salva no disco

    return vetorstore


def analisar_financas(vetorstore, pergunta):

    """
    Executa a busca semântica e analisa os resultados usando um modelo de linguagem.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Configura o pipeline de RAG
    chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=vetorstore.as_retriever(),  # Removido 'search_kwargs={"verbose": True}'
        verbose=True,  # Apenas mantém verbose no pipeline, não no retriever
    )

    resposta = chain.invoke(pergunta)
    return resposta['result']


def salvar_arquivo(resposta, nome_arquivo):
    # Obtém a data e hora atual
    agora = datetime.now()
    
    # Formata a data no formato dia-mes-ano-horario
    data_hora_formatada = agora.strftime("%d%m%Y-%H%M%S")
    
    # Cria o nome do arquivo
    nome_arquivo = f"{nome_arquivo}-{data_hora_formatada}.txt"
    
    try:
        with open(nome_arquivo, "w", encoding="utf-8") as arquivo:
            arquivo.write("📊 Análise Financeira da Empresa\n")
            arquivo.write("=" * 40 + "\n\n")
            arquivo.write(resposta)
    except Exception as error:
        print(f'Erro ao abrir o arquivo: {error}')
    
    print(f"✅ ### Resposta salva com sucesso em: {nome_arquivo}")


if __name__ == "__main__":

    caminho_pdf = "./arquivos/WIZ_04T2025.pdf"
    texto_extraido = extrair_texto_pdf(caminho_pdf)

    # Criar ou carregar o banco vetorial
    base_vetorial = criar_ou_carregar_base_vetorial(texto_extraido)

    resposta = analisar_financas(base_vetorial, prompts.system)

    # Exibir a resposta gerada pelo modelo
    print("\n📊 Análise Financeira da Empresa:")
    salvar_arquivo(resposta, 'ANALISE-')
    print(resposta)
