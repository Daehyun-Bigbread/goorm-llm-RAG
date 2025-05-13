# server/utils/embedding.py
import os
import logging
from typing import List
from langchain.embeddings.base import Embeddings
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

class HuggingFaceInferenceAPIEmbeddings(Embeddings):
    """Hugging Face Inference API를 사용한 임베딩 생성
    
    이 클래스는 기존 임베딩된 데이터 인덱스와 함께 사용되며,
    새로운 쿼리의 임베딩만 생성합니다.
    """
    
    def __init__(self, api_key: str = None, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", provider: str = "novita"):
        """초기화"""
        # 민감한 정보 - 환경 변수에서 로드
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        
        # 비민감 구성 파라미터
        self.model_name = model_name
        self.provider = provider
        
        # 클라이언트 초기화
        self.client = InferenceClient(
            provider=self.provider,
            api_key=self.api_key,
        )
        
        if not self.api_key:
            logger.warning("Hugging Face API 키가 설정되지 않았습니다.")
    
    def _get_embedding(self, text: str) -> List[float]:
        """단일 텍스트의 임베딩 벡터를 가져옵니다."""
        try:
            # 임베딩 생성 API 호출
            embedding = self.client.feature_extraction(
                model=self.model_name,
                text=text,
            )
            return embedding
            
        except Exception as e:
            logger.error(f"임베딩 API 오류: {str(e)}")
            raise ValueError(f"임베딩 API 오류: {str(e)}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 리스트의 임베딩을 생성합니다.
        
        주의: 이 메서드는 일반적으로 기존 임베딩 인덱스가 있으므로 호출되지 않아야 합니다.
        호출된다면 새 문서를 인덱싱하는 과정에서만 사용됩니다.
        """
        logger.warning("embed_documents가 호출되었습니다. 이 메서드는 기존 인덱스가 있으므로 일반적으로 호출되지 않아야 합니다.")
        return [self._get_embedding(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """쿼리 텍스트의 임베딩을 생성합니다.
        
        이 메서드는 검색 시 쿼리 임베딩을 생성하는 데 사용됩니다.
        """
        logger.info(f"쿼리 임베딩 생성: '{text[:30]}...'")
        return self._get_embedding(text)