"""
app/services/llm_service.py
-----------------------------
Step 16  -  Send the grounded prompt to Groq and return the answer.
"""

import logging
from groq import Groq
from app.core.config import settings
from app.services.prompt_service import NO_ANSWER_RESPONSE

logger = logging.getLogger(__name__)


def call_llm(system_prompt: str, user_message: str) -> str:
    """
    Send (system_prompt, user_message) to Groq LLaMA model.
    Returns the model's text response.
    On any error, returns NO_ANSWER_RESPONSE.
    """
    if not system_prompt or not user_message:
        return NO_ANSWER_RESPONSE

    try:
        client = Groq(api_key=settings.LLM_API_KEY)

        response = client.chat.completions.create(
            model=settings.CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.0,
            max_tokens=512,
        )

        answer = response.choices[0].message.content.strip()
        logger.info("LLM answered (%d chars)", len(answer))
        return answer if answer else NO_ANSWER_RESPONSE

    except Exception as e:
        logger.error("LLM error: %s", e)
        if "429" in str(e) or "rate" in str(e).lower():
            return "Rate limit reached. Please wait a moment and try again."
        return NO_ANSWER_RESPONSE