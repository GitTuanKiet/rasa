import json
import logging
import os
from typing import Optional, Text

import google.generativeai as genai

logger = logging.getLogger(__name__)

class GenerativeAI:
    def __init__(
        self,
        completions_model=os.environ.get(
            "GENERATIVEAI_COMPLETIONS_MODEL", "gemini-pro"
        ),
        completions_temperature=int(
            os.environ.get("GENERATIVEAI_COMPLETIONS_TEMPERATURE", 0)
        ),
        completions_max_tokens=int(
            os.environ.get("GENERATIVEAI_COMPLETIONS_MAX_TOKENS", 100)
        ),
        completions_api_key=os.environ.get("GENERATIVEAI_COMPLETIONS_API_KEY", None),
    ):
        self.completions_model = completions_model
        self.completions_temperature = completions_temperature
        self.completions_max_tokens = completions_max_tokens
        self.completions_api_key = completions_api_key

    @staticmethod
    def _extract_text_response(text: Text) -> Optional[Text]:
        try:
            logger.info(f"LLM response: \n{text}\n")

            response_json = json.loads(text)
            return response_json.get("answer", None)
        except Exception as e:
            logger.exception(f"Error occurred while extracting the LLM response. {e}")
            return None

    def get_text_completion(self, prompt: Text) -> Optional[Text]:
        logger.info(f"LLM prompt: \n{prompt}\n")

        genai.configure(api_key=self.completions_api_key)
        model = genai.GenerativeModel(self.completions_model)

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=self.completions_temperature,
                max_output_tokens=self.completions_max_tokens,
            ),
        )

        return self._extract_text_response(text=response.text)
