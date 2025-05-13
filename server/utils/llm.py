# server/utils/llm.py
import os
import logging
from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

# Hugging Face Inference API
class HuggingFaceInferenceAPI(LLM):
    
    api_key: str = None
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    temperature: float = 0.2
    max_tokens: int = 512
    top_p: float = 0.8
    provider: str = "novita"
    
    @property
    def _llm_type(self) -> str:
        return "huggingface_inference_api"
    
    # LLM API 호출
    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> str:
        api_key = self.api_key or os.getenv("HUGGINGFACE_API_KEY")
        
        try:
            # InferenceClient 초기화
            client = InferenceClient(
                provider=self.provider,
                api_key=api_key,
            )
            
            # 채팅 완성 요청
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p
            )
            
            # 응답 추출
            response = completion.choices[0].message.content
            logger.info(f"LLM 응답 생성 완료: {len(response)} 글자")
            
            return response
            
        except Exception as e:
            logger.error(f"LLM API 호출 오류: {str(e)}")
            raise ValueError(f"LLM API 호출 오류: {str(e)}")
    
    # 모델 식별 파라미터
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "provider": self.provider
        }