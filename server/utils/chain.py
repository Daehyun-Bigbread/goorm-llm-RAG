# server/utils/chain.py
import os
import logging
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import config

from .embedding import HuggingFaceInferenceAPIEmbeddings
from .llm import HuggingFaceInferenceAPI
from .document import load_documents_and_index, check_faiss_index_exists

logger = logging.getLogger(__name__)

def initialize_rag_pipeline() -> RetrievalQA:
    """RAG 파이프라인을 초기화합니다."""
    try:
        # 임베딩 모델 초기화 (쿼리 임베딩에만 사용)
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
        
        # FAISS 인덱스 확인
        if check_faiss_index_exists():
            # 기존 FAISS 인덱스 로드
            logger.info(f"기존 FAISS 인덱스 로드: {config.FAISS_INDEX_PATH}")
            vectorstore = FAISS.load_local(
                folder_path=str(config.DATA_DIR),
                embeddings=embeddings,
                index_name="document_index"
            )
        else:
            # 이미 존재해야 하는 인덱스가 없으면 오류 발생
            logger.error(f"FAISS 인덱스를 찾을 수 없습니다: {config.FAISS_INDEX_PATH}")
            raise FileNotFoundError(f"FAISS 인덱스 파일이 존재하지 않습니다: {config.FAISS_INDEX_PATH}")
        
        # 검색기 생성
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": int(os.getenv("RETRIEVER_TOP_K", 3))}
        )
        
        # LLM 초기화 - 새로운 API 클라이언트 사용
        llm = HuggingFaceInferenceAPI(
            temperature=float(os.getenv("TEMPERATURE", 0.2)),
            max_tokens=int(os.getenv("MAX_NEW_TOKENS", 512)),
            top_p=float(os.getenv("TOP_P", 0.95)),
            model_name=os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        )
        
        # 프롬프트 템플릿 생성
        prompt_template = """다음 문맥을 바탕으로 질문에 답하세요. 문맥에 관련 정보가 없으면 "주어진 문맥에서 답을 찾을 수 없습니다."라고 응답하세요.

문맥: {context}

질문: {question}

답변:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # QA 체인 생성
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        logger.info("RAG 파이프라인이 성공적으로 초기화되었습니다.")
        return qa_chain
    
    except Exception as e:
        logger.error(f"RAG 파이프라인 초기화 오류: {str(e)}")
        raise RuntimeError(f"RAG 파이프라인 초기화 중 오류가 발생했습니다: {str(e)}")