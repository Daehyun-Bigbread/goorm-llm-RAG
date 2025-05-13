# server/utils/embedding.py
import os
import logging
import requests
from typing import List
from langchain.embeddings.base import Embeddings

logger = logging.getLogger(__name__)

class HuggingFaceInferenceAPIEmbeddings(Embeddings):
    """Hugging Face Inference API를 사용한 임베딩 생성
    
    이 클래스는 기존 임베딩된 데이터 인덱스와 함께 사용되며,
    새로운 쿼리의 임베딩만 생성합니다.
    """
    
    def __init__(self, api_key: str = None, model_name: str = "nlpai-lab/KURE-v1"):
        """초기화"""
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}"
        
        if not self.api_key:
            logger.warning("Hugging Face API 키가 설정되지 않았습니다.")
    
    def _get_embedding(self, text: str) -> List[float]:
        """단일 텍스트의 임베딩 벡터를 가져옵니다."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            self.api_url, 
            headers=headers, 
            json={"inputs": text, "options": {"wait_for_model": True}}
        )
        
        if response.status_code != 200:
            logger.error(f"임베딩 API 오류: {response.text}")
            raise ValueError(f"임베딩 API 오류: {response.text}")
        
        return response.json()
    
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