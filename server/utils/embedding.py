# server/utils/embedding.py
import logging
from typing import List
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class SimpleLocalEmbeddings(Embeddings):
    """간단한 로컬 SentenceTransformer 임베딩"""
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """초기화"""
        self.model_name = model_name
        
        # 모델 로드
        logger.info(f"SentenceTransformer 모델 로드 중: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info(f"모델 로드 완료: 차원 = {self.model.get_sentence_embedding_dimension()}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 리스트의 임베딩을 생성합니다."""
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """쿼리 텍스트의 임베딩을 생성합니다."""
        logger.info(f"쿼리 임베딩 생성: '{text[:30]}...'")
        return self.model.encode(text).tolist()