# 워키피디아 데이터 기반 LLM 서비스

위키피디아 데이터(KorQuAD)를 활용하여 사용자의 질문에 대한 답변을 생성하는 LLM 서버입니다. RAG(Retrieval-Augmented Generation) 아키텍처를 기반으로 구현되어, 질문에 관련되어 임베딩된 KorQuAD 1.0 Dataset(위키피디아 데이터셋)문서를 검색하고 이를 바탕으로 정확한 응답을 생성합니다.

### 주요 기능
- 사용자 질문에 대한 관련 KorQuAD 1.0 Dataset(위키피디아 데이터셋)에서 검색
- 검색된 KorQuAD 1.0 Dataset(위키피디아 데이터셋) 기반 정확한 응답 생성
- 응답 생성 시 참조한 KorQuAD 1.0 Dataset(위키피디아 데이터셋) 문서 출처 제공
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
# .env 파일을 최상단 디릭토리에 생성
"OPENAI_API_KEY=your_openai_api_key"
"HUGGINGFACE_API_KEY=your_huggingface_api_key"
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

## API 사용
API는 Swagger UI를 통해 문서화되어 있습니다.
- 서버 실행 후 http://localhost:8000/docs에서 API 문서를 확인할 수 있습니다.

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

### Docker를 이용한 설치 및 실행

```bash
# Docker 이미지 빌드
docker build -t llm-rag-wiki-service .

# Docker 컨테이너 실행 (환경 변수 전달)
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_openai_api_key \
  -e HUGGINGFACE_API_KEY=your_huggingface_api_key \
  llm-rag-wiki-service
```

### 응답 예시
```json
{
  "retrieved_document_id": 6559027,
  "retrieved_document": "천안함 사건이 일어나기 직전 북한 잠수정과 해안포의 심상치 않은 움직임이 포착됐는데도, 우리 군이 적절한 대응조치를 취하지 않았다는 논란이 있다. 민주당 신학용의원은 천안함 사건 당일 아침 해군 2함대가 전파한 군 문자정보망 교신 내역을 공개하며 2~3일 전부터 북한 연어급 잠수정 한척과 이를 지원하는 배 6척이 기지에서 출동한 뒤 모습을 보이지 않고, 천안함 침몰 직전 시점에는 북한 해안포 10문이 모습을 드러냈는데도 이렇게 정보가 있음에도 불구하고 전투태세 및 경계태세는 발령되지 않은 점을 문제삼았다. 국방부장관은 \"오늘날 지나고 나서 다 이 사건이 북한의 잠수함 어뢰공격이었구나 하는 것을 아니까 오늘날 우리가 이렇게 얘기할 수 있습니다만 그 당시에는 쉽지 않았다\"며, 그것을 침투나 도발 징후로 인정하지를 않았다고 해명했다. 민주당 박지원 원내대표는 \"많은 우리 장병들이 이로 인해 목숨을 잃었지만 실질적인 최고 책임자 김태영 국방장관은 오늘까지도 국방장관을 엔조이하고 있다\"며 국방부장관을 해임하고 군법회의에 회부해야 한다고 주장했다. 이에 한나라당은 \"그로 인해 장병들이 목숨을 잃었다\" 는 표현은 천안함 사건이 북한의 폭침에 의한 것임을 시인한 것” 이라며 \"민주당은 지금부터라도 북한에 천안함 책임을 강하게 물어야 책임 있는 공당이 될 수 있다\" 고 했다. 민주당 손학규 대표는 “우리는 한국의 공식 입장과 발표를 신뢰한다는 자세를 갖고 있다”면서도 “ 여러 의혹이 정부에 의해 분명히 밝혀지지 않는 것도 사실인 만큼 문제 제기는 야당의 당연한 임무”라고 말했다.",
  "question": "천안함 사건이 언제 일어났는지 알려줘",
  "answers": "천안함 사건은 2010년 3월 26일에 백령도 근처 해상에서 대한민국 해군의 초계함인 PCC-772 천안이 피격되어 침몰한 사건이다."
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
- **FastAPI Server**: RESTful API 엔드포인트 제공, 미들웨어로 로깅 및 예외 처리
- **RAG Pipeline**: LangChain 기반 질의응답 Pipeline
- **Vector DB**: FAISS 인덱스를 사용한 유사도 검색
- **LLM Model(Llama-3.1-8B-Instruct)**: 오픈소스 LLM 모델 (Llama-3.1-8B-Instruct)을 활용한 텍스트 생성 (from Huggingface Inference API)

### 설계 원칙
- **RAG**: KorQuAD 1.0 Dataset(위키피디아 데이터셋)을 활용하여 LLM의 응답에 활용
- **모듈화**: Chain, Embedding, LLM Inference 등 각 컴포넌트 독립적 구현
- **로깅**: 구조화된 로깅으로 문제 진단 및 모니터링 용이

### 기술 스택
- **언어**: Python 3.8+
- **API Framework**: FastAPI
- **RAG Framework**: LangChain
- **Vector DB**: FAISS
- **LLM**: Huggingface 오픈소스 모델 (Llama-3.1-8B-Instruct)
- ** 사용자 Query Embedding**: OpenAI 임베딩 API (text-embedding-ada-002)

### 데이터 처리 파이프라인
1. **Preprocess(전처리)**: KorQuAD 1.0 Dataset(위키피디아 데이터셋)에서 문서 추출, 전처리, 임베딩
2. **Embedding(임베딩)**: OpenAI Embedding API(text-embedding-ada-002)를 사용해 사용자 쿼리 임베딩 생성
3. **Vector Indexing**: FAISS를 통한 Vector Index 구축
4. **검색**: 사용자 쿼리(질문) 임베딩 기반 유사 문서 검색 (MMR 알고리즘 사용)
5. **Question & Answering**: 검색된 문서 컨텍스트를 바탕으로 LLM(Llama-3.1-8B-Instruct)을 통한 답변 생성
