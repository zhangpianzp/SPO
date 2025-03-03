import asyncio
import re
from enum import Enum
from typing import Any, List, Optional

from metagpt.configs.models_config import ModelsConfig
from metagpt.llm import LLM
from metagpt.logs import logger


class RequestType(Enum):
    OPTIMIZE = "optimize"
    EVALUATE = "evaluate"
    EXECUTE = "execute"


class SPO_LLM:
    _instance: Optional["SPO_LLM"] = None

    def __init__(
        self,
        config_path: Optional[str] = None,
        optimize_kwargs: Optional[dict] = None,
        evaluate_kwargs: Optional[dict] = None,
        execute_kwargs: Optional[dict] = None,
    ) -> None:
        self.evaluate_llm = LLM(
            llm_config=self._load_llm_config(config_path, evaluate_kwargs))
        self.optimize_llm = LLM(
            llm_config=self._load_llm_config(config_path, optimize_kwargs))
        self.execute_llm = LLM(
            llm_config=self._load_llm_config(config_path, execute_kwargs))

    def _load_llm_config(self, config_path: str, kwargs: dict) -> Any:
        model = kwargs.get("model")
        if not model:
            raise ValueError("'model' parameter is required")

        try:
            if config_path:
                models_config = ModelsConfig.from_home(config_path)
            else:
                models_config = ModelsConfig.default()

            model_config = models_config.get(model)
            if model_config is None:
                raise ValueError(f"Model '{model}' not found in configuration")

            config = model_config.model_copy()

            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            return config

        except AttributeError:
            raise ValueError(f"Model '{model}' not found in configuration")
        except Exception as e:
            raise ValueError(
                f"Error loading configuration for model '{model}': {str(e)}")

    async def responser(self, request_type: RequestType, messages: List[dict]) -> str:
        llm_mapping = {
            RequestType.OPTIMIZE: self.optimize_llm,
            RequestType.EVALUATE: self.evaluate_llm,
            RequestType.EXECUTE: self.execute_llm,
        }

        llm = llm_mapping.get(request_type)
        if not llm:
            raise ValueError(
                f"Invalid request type. Valid types: {', '.join([t.value for t in RequestType])}")

        response = await llm.acompletion(messages)
        return response.choices[0].message.content

    @classmethod
    def initialize(cls, config_path: str, optimize_kwargs: dict, evaluate_kwargs: dict, execute_kwargs: dict) -> None:
        """Initialize the global instance"""
        cls._instance = cls(config_path, optimize_kwargs,
                            evaluate_kwargs, execute_kwargs)

    @classmethod
    def get_instance(cls) -> "SPO_LLM":
        """Get the global instance"""
        if cls._instance is None:
            raise RuntimeError(
                "SPO_LLM not initialized. Call initialize() first.")
        return cls._instance


def extract_content(xml_string: str, tag: str) -> Optional[str]:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, xml_string, re.DOTALL)
    return match.group(1).strip() if match else None


async def main():
    # test LLM
    SPO_LLM.initialize(
        optimize_kwargs={"model": "gpt-4o", "temperature": 0.7},
        evaluate_kwargs={"model": "gpt-4o-mini", "temperature": 0.3},
        execute_kwargs={"model": "gpt-4o-mini", "temperature": 0.3},
    )

    llm = SPO_LLM.get_instance()

    # test messages
    hello_msg = [{"role": "user", "content": "hello"}]
    response = await llm.responser(request_type=RequestType.EXECUTE, messages=hello_msg)
    logger(f"AI: {response}")
    response = await llm.responser(request_type=RequestType.OPTIMIZE, messages=hello_msg)
    logger(f"AI: {response}")
    response = await llm.responser(request_type=RequestType.EVALUATE, messages=hello_msg)
    logger(f"AI: {response}")


if __name__ == "__main__":
    asyncio.run(main())
