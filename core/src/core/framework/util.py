import logging
from typing import Any, Dict, List
from uuid import UUID
from langchain.schema import LLMResult
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class OllamaCallbacks(BaseCallbackHandler):
    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        messages = "\n".join(map(str, messages))
        logger.debug(f"[[Messages]]\n{messages}")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        logger.debug(f"{response}")
