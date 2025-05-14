# 워키피디아 데이터 기반 LLM 서비스

위키피디아 데이터(KorQuAD)를 활용하여 사용자의 질문에 대한 답변을 생성하는 LLM 서버입니다. RAG(Retrieval-Augmented Generation) 아키텍처를 기반으로 구현되어, 질문에 관련되어 임베딩된 위키피디아 문서를 검색하고 이를 바탕으로 정확한 응답을 생성합니다.

### 주요 기능
- 사용자 질문에 대한 관련 위키피디아 문서 검색
- 검색된 문서 기반 정확한 응답 생성
- 응답 생성 시 참조한 위키피디아 문서 출처 제공
- 질의 범위 제한 메커니즘 (관련 없는 답변 출력 제한)

## 프로젝트 구조
```
.
├── config.py                   # 환경 설정 및 경로 관리
├── data                        # 데이터 저장 디렉토리
│   ├── document_index.faiss    # FAISS 벡터 인덱스
│   ├── document_index.pkl      # 인덱스 메타데이터
│   └── documents.json          # 전처리된 위키피디아 문서
├── data_preprocess             # 데이터 전처리 스크립트
│   └── data_preprocess_openai.ipynb
├── main.py                     # API 서버 진입점
└── server                      # 서버 코드 디렉토리
    ├── routers                 # API 라우터
    │   └── question_answer.py  # Q&A API 엔드포인트
    ├── schemas                 # API 요청/응답 스키마
    │   └── schema.py
    └── utils                   # 유틸리티 기능
        ├── chain.py            # RAG 체인 구현
        ├── document.py         # 문서 처리
        ├── embedding.py        # 임베딩 처리
        └── llm.py              # LLM 모델 관리
```

## 설치 및 실행 방법

### 요구 사항
- Python 3.8 이상
- 충분한 메모리(최소 8GB 권장)
- GPU 환경 권장(없을 경우 CPU에서도 동작 가능)
- API 키: OpenAI(임베딩용), Hugging Face(LLM 추론용)

### 설치 방법
```bash
# 저장소 클론
git clone https://github.com/username/wikipedia-llm-service.git
cd wikipedia-llm-service

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 환경 변수 설정
```bash
# .env 파일 생성
echo "OPENAI_API_KEY=your_openai_api_key" > .env
echo "HUGGINGFACE_API_KEY=your_huggingface_api_key" >> .env
```

### 실행 방법
```bash
# 데이터 전처리 및 임베딩 생성
# data_preprocess/data_preprocess_openai.ipynb 노트북 실행

# API 서버 실행
python main.py

# 또는 uvicorn 직접 실행
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API 사용 예시
API는 Swagger UI를 통해 문서화되어 있습니다. 서버 실행 후 http://localhost:8000/docs에서 API 문서를 확인할 수 있습니다.

### API 엔드포인트
| 엔드포인트 | 메소드 | 설명 |
|------------|--------|------|
| `/api/llmserver/query` | POST | 질문에 대한 답변 생성 |
| `/health` | GET | 서버 상태 확인 |

### Swagger에서 테스트 방법
1. 서버를 실행한 후 브라우저에서 `http://localhost:8000/docs` 접속
2. '/api/llmserver/query' 엔드포인트 펼치기
3. 'Try it out' 버튼 클릭
4. 요청 본문에 질문 입력:
   ```json
   {
     "question": "대한민국의 수도는 어디인가요?"
   }
   ```
5. 'Execute' 버튼 클릭하여 결과 확인

### curl 요청 예시 (Postman)
```bash
curl -X 'POST' \
  'http://localhost:8000/api/llmserver/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "대한민국의 수도는 어디인가요?"
}'
```

### 요청 예시 (Python)
```python
import requests
import json

# 질문 API 요청
response = requests.post(
    "http://localhost:8000/api/llmserver/query",
    headers={"Content-Type": "application/json"},
    json={"question": "대한민국의 수도는 어디인가요?"}
)

print(json.dumps(response.json(), indent=2, ensure_ascii=False))
```

### 요청 예시 (JavaScript)
```javascript
fetch('http://localhost:8000/api/llmserver/query', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    question: '대한민국의 수도는 어디인가요?'
  })
})
  .then(response => response.json())
  .then(data => console.log(data));
```

### 응답 예시
```json
{
  "retrieved_document_id": 42,
  "retrieved_document": "대한민국(大韓民國, 영어: Republic of Korea, 문화어: 조선민주주의인민공화국)은 동아시아의 한반도 남부에 위치한 국가이다. 서쪽으로는 서해(황해)를 사이에 두고 중화인민공화국과 마주하고 있으며, 동쪽으로는 동해(일본해)를 사이에 두고 일본과 마주하고 있다. 북쪽으로는 조선민주주의인민공화국(북한)과 국경을 접하고 있다. 대한민국의 수도는 서울특별시이다. 면적은 100,210 km²이다.",
  "question": "대한민국의 수도는 어디인가요?",
  "answers": "대한민국의 수도는 서울특별시입니다."
}
```

## 구현 아키텍처 및 접근 방식

### 아키텍처 개요
```
┌─────────────┐      ┌──────────────────┐      ┌───────────────┐
│  사용자 요청  │─────▶│   FastAPI 서버    │─────▶│  벡터 데이터베이스│
└─────────────┘      └──────────────────┘      └───────┬───────┘
                           │       ▲                   │
                           ▼       │                   ▼
                     ┌──────────────────┐      ┌───────────────┐
                     │  RAG 파이프라인    │◀─────│ 임베딩된 문서들 │
                     └──────────────────┘      └───────────────┘
                           │       ▲
                           ▼       │
                     ┌──────────────────┐
                     │     LLM 모델     │
                     └──────────────────┘
```

### 주요 컴포넌트
- **FastAPI 서버**: RESTful API 엔드포인트 제공, 미들웨어로 로깅 및 예외 처리
- **RAG 파이프라인**: LangChain 기반 질의응답 파이프라인
- **벡터 데이터베이스**: FAISS 인덱스를 사용한 효율적인 유사도 검색
- **LLM 모델**: Huggingface 모델을 활용한 텍스트 생성

### 설계 원칙
- **RAG 아키텍처**: 기존 지식(위키피디아)을 활용하여 LLM의 응답 품질 향상
- **모듈화 설계**: 검색, 임베딩, LLM 추론 등 각 컴포넌트 독립적 구현으로 유지보수성 확보
- **효율적인 로깅**: 구조화된 로깅으로 문제 진단 및 모니터링 용이
- **예외 처리**: 세분화된 예외 처리로 안정적인 서비스 제공

### 기술 스택
- **언어**: Python 3.8+
- **웹 프레임워크**: FastAPI
- **RAG 프레임워크**: LangChain
- **벡터 데이터베이스**: FAISS
- **LLM**: Huggingface 오픈소스 모델 (Llama-3.1-8B-Instruct)
- **임베딩**: OpenAI 임베딩 API (text-embedding-ada-002)

### 데이터 처리 파이프라인
1. **전처리**: KorQuAD 데이터셋에서 위키피디아 문서 추출 및 정제
2. **임베딩**: OpenAI API를 사용해 문서 텍스트 임베딩 생성
3. **인덱싱**: FAISS를 통한 벡터 인덱스 구축
4. **검색**: 사용자 질문 임베딩 기반 유사 문서 검색 (MMR 알고리즘 사용)
5. **생성**: 검색된 문서 컨텍스트를 바탕으로 LLM을 통한 답변 생성
