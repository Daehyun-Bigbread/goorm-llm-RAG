# server/utils/embedding.py
import logging
from typing import List
from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings
import os

logger = logging.getLogger(__name__)

class SimpleLocalEmbeddings(Embeddings):
    """OpenAI 임베딩 API를 사용한 쿼리 임베딩 클래스"""
    
    def __init__(self, model_name="text-embedding-ada-002"):
        """초기화"""
        self.model_name = model_name
        
        # OpenAI API 키 확인
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OpenAI API 키가 설정되지 않았습니다. 'OPENAI_API_KEY' 환경 변수를 설정하세요.")
        
        # OpenAI 임베딩 모델 초기화
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=self.api_key
        )
        
        logger.info(f"OpenAI 임베딩 모델 초기화 완료: {model_name}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 리스트의 임베딩을 생성합니다."""
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """쿼리 텍스트의 임베딩을 생성합니다."""
        logger.info(f"쿼리 임베딩 생성: '{text[:30]}...'")
        return self.embeddings.embed_query(text)