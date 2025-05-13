# server/utils/chain.py
import os
import logging
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import config

from .embedding import SimpleLocalEmbeddings
from .llm import HuggingFaceInferenceAPI
from .document import load_documents_and_index, check_faiss_index_exists

logger = logging.getLogger(__name__)

def initialize_rag_pipeline() -> RetrievalQA:
    """RAG 파이프라인을 초기화합니다."""
    try:
        # 로컬 임베딩 모델 초기화 (문서 임베딩은 이미 생성된 index 사용)
        embeddings = SimpleLocalEmbeddings()

        # FAISS 인덱스 확인
        if check_faiss_index_exists():
            # 기존 FAISS 인덱스 로드
            logger.info(f"기존 FAISS 인덱스 로드: {config.FAISS_INDEX_PATH}")
            vectorstore = FAISS.load_local(
                folder_path=str(config.DATA_DIR),
                embeddings=embeddings,
                index_name="document_index",
                allow_dangerous_deserialization=True
            )
        else:
            # 이미 존재해야 하는 인덱스가 없으면 오류 발생
            logger.error(f"FAISS 인덱스를 찾을 수 없습니다: {config.FAISS_INDEX_PATH}")
            raise FileNotFoundError(f"FAISS 인덱스 파일이 존재하지 않습니다: {config.FAISS_INDEX_PATH}")
        
        # 검색기 생성
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}  # 더 많은 문서 검색
        )
        
        # LLM 초기화 - 새로운 API 클라이언트 사용
        llm = HuggingFaceInferenceAPI(
            temperature=float(os.getenv("TEMPERATURE", 0.0)),
            max_tokens=int(os.getenv("MAX_NEW_TOKENS", 512)),
            top_p=float(os.getenv("TOP_P", 1.0)),
            model_name=os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        )
        
        prompt_template = """당신은 제공된 문맥에만 기반하여 정확하게 답변하는 시스템입니다.

엄격한 규칙:
1. 오직 제공된 문맥에 명시적으로 포함된 정보만 사용하세요.
2. 문맥에 명확하게 답이 없는 경우 반드시 "주어진 문맥에서 답을 찾을 수 없습니다."라고만 응답하세요.
3. 절대로 스스로 정보를 생성하거나 추측하지 마세요.
4. 짧고 직접적으로 답변하세요.
5. 문맥에 없는 연도, 날짜, 이름 또는 수치를 절대 포함하지 마세요.

문맥:
{context}

질문: {question}

답변 (반드시 위 규칙을 따르고, 문맥에 없는 정보는 절대 포함하지 마세요):"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # QA 체인 생성 (stuff 방식)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # map_reduce 대신 stuff 사용
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        logger.info("RAG 파이프라인이 성공적으로 초기화되었습니다.")
        return qa_chain
    
    except Exception as e:
        logger.error(f"RAG 파이프라인 초기화 오류: {str(e)}")
        raise RuntimeError(f"RAG 파이프라인 초기화 중 오류가 발생했습니다: {str(e)}")