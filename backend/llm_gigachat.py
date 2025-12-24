"""
Клиент GigaChat через langchain-gigachat: генерация ответа и обёртка для Ragas.
"""

from os import getenv
from typing import Optional

from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Any

try:
    from ragas.llms import LangchainLLMWrapper
except Exception:
    LangchainLLMWrapper = None


def _as_bool(v: Optional[str], default: bool) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "t", "yes", "y"}


def _client() -> GigaChat:
    credentials = (getenv("GIGACHAT_CREDENTIALS") or "").strip()
    if not credentials:
        raise RuntimeError("GIGACHAT_CREDENTIALS не задан")
    scope = (getenv("GIGACHAT_SCOPE") or "GIGACHAT_API_PERS").strip()
    model = (getenv("GIGACHAT_MODEL") or "GigaChat-2").strip()
    verify_ssl = _as_bool(getenv("GIGACHAT_VERIFY_SSL"), False)
    streaming = _as_bool(getenv("GIGACHAT_STREAMING"), False)
    return GigaChat(
        credentials=credentials,
        scope=scope,
        model=model,
        verify_ssl_certs=verify_ssl,
        streaming=streaming,
        temperature=0.0,
    )


def generate(system: str, user: str) -> str:
    """
    Возвращает текст LLM-ответа по заданным системному и пользовательскому сообщениям.
    """
    giga = _client()
    msgs = [SystemMessage(content=system), HumanMessage(content=user)]
    res = giga.invoke(msgs)
    return res.content or ""


def ragas_llm() -> Any:
    """
    Возвращает LLM-адаптер для Ragas, использующий GigaChat.
    """
    if LangchainLLMWrapper is None:
        raise RuntimeError(
            "LangchainLLMWrapper недоступен. Проверьте версию ragas.")
    return LangchainLLMWrapper(_client())
