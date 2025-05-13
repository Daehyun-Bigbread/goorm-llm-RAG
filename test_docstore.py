# test_docstore.py
import pickle
from pathlib import Path

# 데이터 경로
data_dir = Path("/Users/daehyunkim/Desktop/goorm-llm-RAG/data")
pkl_path = data_dir / "document_index.pkl"

# 메타데이터 로드
with open(pkl_path, "rb") as f:
    metadata = pickle.load(f)

# 메타데이터 구조 확인
print(f"메타데이터 타입: {type(metadata)}")
print(f"튜플 길이: {len(metadata)}")

# InMemoryDocstore 확인
docstore = metadata[0]
print(f"Docstore 타입: {type(docstore)}")

# docstore에서 문서 접근 방법 확인
if hasattr(docstore, "_dict"):
    print(f"Docstore 문서 수: {len(docstore._dict)}")
    
    # 일부 문서 ID 확인
    print("\n문서 ID 샘플:")
    doc_ids = list(docstore._dict.keys())[:5]
    for doc_id in doc_ids:
        print(f"  {doc_id}")
    
    # 문서 내용 샘플 확인
    print("\n문서 내용 샘플:")
    for doc_id in doc_ids:
        doc = docstore._dict[doc_id]
        print(f"ID: {doc_id}")
        print(f"내용: {doc.page_content[:100]}...")
        print(f"메타데이터: {doc.metadata}\n")
    
    # 유엔 관련 문서 검색
    print("\n유엔 키워드 검색:")
    found = 0
    for doc_id, doc in docstore._dict.items():
        if "유엔" in doc.page_content or "UN" in doc.page_content or "국제 연합" in doc.page_content:
            print(f"유엔 관련 문서 발견 - ID: {doc_id}")
            print(f"내용: {doc.page_content[:200]}...")
            print()
            found += 1
            if found >= 5:
                print("더 많은 문서가 있을 수 있습니다...")
                break
    
    if found == 0:
        print("유엔 관련 문서를 찾을 수 없습니다.")