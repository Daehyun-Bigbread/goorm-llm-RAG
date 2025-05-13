# server/utils/llm.py
import os
import logging
import requests
from typing import List, Dict, Any, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_community.llms.utils import enforce_stop_tokens

logger = logging.getLogger(__name__)

class HuggingFaceInferenceAPILLM(LLM):
    
    api_key: str = None
    model_name: str = "beomi/KoAlpaca-Polyglot-5.8B"
    temperature: float = 0.2
    max_new_tokens: int = 512
    top_p: float = 0.95
    
    @property
    def _llm_type(self) -> str:
        return "huggingface_inference_api"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """LLM API 호출"""
        api_key = self.api_key or os.getenv("HUGGINGFACE_API_KEY")
        api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 생성 파라미터
        parameters = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": True,
            "return_full_text": False
        }
        
        # API 요청
        payload = {
            "inputs": prompt,
            "parameters": parameters,
            "options": {"wait_for_model": True}
        }
        
        response = requests.post(api_url, headers=headers, json=payload)
        
        if response.status_code != 200:
            logger.error(f"LLM API 오류: {response.text}")
            raise ValueError(f"LLM API 오류: {response.text}")
        
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            text = result[0].get("generated_text", "")
        elif isinstance(result, dict):
            text = result.get("generated_text", "")
        else:
            text = str(result)
        
        # 중단 토큰 처리
        if stop:
            text = enforce_stop_tokens(text, stop)
        
        return text