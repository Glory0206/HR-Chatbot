# rag_chain.py

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# RAG 파이프라인을 실행하는 함수
def get_rag_chain_answer(document_text: str, question: str) -> str:
    # 텍스트를 Document 객체로 변환
    docs = [Document(page_content=document_text)]

    # 텍스트 분할 (Split)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)

    # 임베딩 모델 로드 (한국어 모델 사용)
    model_name = "jhgan/ko-sbert-nli"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # 벡터 저장소 생성 (Store)
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # 검색기 생성 (Retrieve)
    retriever = vectorstore.as_retriever()

    # 프롬프트 템플릿 정의
    template = """
    당신은 사내 HR 규정 전문가입니다. 제공된 내용을 바탕으로 사용자의 질문에 답변해 주세요.
    내용에 없는 정보는 답변하지 마세요.

    [내용]:
    {context}

    [질문]:
    {question}

    [답변]:
    """
    prompt = PromptTemplate.from_template(template)

    # LLM 모델 정의
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    # RAG 체인 구성 및 실행 (Generate)
    rag_chain = prompt | llm | StrOutputParser()
    
    # 검색된 문서와 질문을 함께 체인에 전달
    retrieved_docs = retriever.invoke(question)
    answer = rag_chain.invoke({"context": retrieved_docs, "question": question})

    return answer