# server/utils/document.py
import json
import pickle
import logging
from typing import List, Dict, Tuple
from langchain.schema import Document
import config

logger = logging.getLogger(__name__)

def load_documents_and_index() -> Tuple[List[Document], Dict]:
    """문서 데이터와 인덱스 정보를 로드합니다."""
    try:
        # 문서 데이터 로드
        with open(config.DOCUMENTS_PATH, "r", encoding="utf-8") as f:
            raw_docs = json.load(f)
        
        # ID-인덱스 매핑 로드
        with open(config.ID_TO_INDEX_PATH, "rb") as f:
            id_to_index = pickle.load(f)
        
        # LangChain Document 객체로 변환
        documents = []
        for doc in raw_docs:
            documents.append(
                Document(
                    page_content=doc["context"],
                    metadata={
                        "id": doc["id"],
                        "title": doc["title"]
                    }
                )
            )
        
        logger.info(f"{len(documents)}개의 문서를 로드했습니다.")
        return documents, id_to_index
    
    except Exception as e:
        logger.error(f"문서 및 인덱스 로드 오류: {str(e)}")
        raise RuntimeError(f"문서 및 인덱스 로드 중 오류가 발생했습니다: {str(e)}")

def check_faiss_index_exists() -> bool:
    """FAISS 인덱스 파일이 존재하는지 확인합니다."""
    return config.FAISS_INDEX_PATH.exists()