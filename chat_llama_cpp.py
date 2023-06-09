import logging
from langchain.chat_models.base import BaseChatModel
from typing import Any, Dict, Generator, List, Optional
from pydantic import Field, root_validator
from langchain.callbacks.manager import (
    # AsyncCallbackManager,
    AsyncCallbackManagerForLLMRun,
    # CallbackManager,
    CallbackManagerForLLMRun,
    # Callbacks,
)
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatResult,
    # HumanMessage,
    # LLMResult,
    # PromptValue,
)
from llama_cpp import ChatCompletionMessage


logger = logging.getLogger(__name__)


class ChatLlamaCpp(BaseChatModel):
    """Wrapper around the llama.cpp model.

    To use, you should have the llama-cpp-python library installed, and provide the
    path to the Llama model as a named parameter to the constructor.
    Check out:

    Example:
        .. code-block:: python

            from langchain.chat_models import ChatLlamaCpp
            llm = ChatLlamaCpp(model_path="/path/to/llama/model")
    """

    client: Any  #: :meta private:
    model_path: str
    """The path to the Llama model file."""

    lora_base: Optional[str] = None
    """The path to the Llama LoRA base model."""

    lora_path: Optional[str] = None
    """The path to the Llama LoRA. If None, no LoRa is loaded."""

    n_ctx: int = Field(512, alias="n_ctx")
    """Token context window."""

    n_parts: int = Field(-1, alias="n_parts")
    """Number of parts to split the model into.
    If -1, the number of parts is automatically determined."""

    seed: int = Field(-1, alias="seed")
    """Seed. If -1, a random seed is used."""

    f16_kv: bool = Field(True, alias="f16_kv")
    """Use half-precision for key/value cache."""

    logits_all: bool = Field(False, alias="logits_all")
    """Return logits for all tokens, not just the last token."""

    vocab_only: bool = Field(False, alias="vocab_only")
    """Only load the vocabulary, no weights."""

    use_mlock: bool = Field(False, alias="use_mlock")
    """Force system to keep model in RAM."""

    n_threads: Optional[int] = Field(None, alias="n_threads")
    """Number of threads to use.
    If None, the number of threads is automatically determined."""

    n_batch: Optional[int] = Field(8, alias="n_batch")
    """Number of tokens to process in parallel.
    Should be a number between 1 and n_ctx."""

    n_gpu_layers: Optional[int] = Field(None, alias="n_gpu_layers")
    """Number of layers to be loaded into gpu memory. Default None."""

    suffix: Optional[str] = Field(None)
    """A suffix to append to the generated text. If None, no suffix is appended."""

    max_tokens: Optional[int] = 256
    """The maximum number of tokens to generate."""

    temperature: Optional[float] = 0.8
    """The temperature to use for sampling."""

    top_p: Optional[float] = 0.95
    """The top-p value to use for sampling."""

    logprobs: Optional[int] = Field(None)
    """The number of logprobs to return. If None, no logprobs are returned."""

    echo: Optional[bool] = False
    """Whether to echo the prompt."""

    stop: Optional[List[str]] = []
    """A list of strings to stop generation when encountered."""

    repeat_penalty: Optional[float] = 1.1
    """The penalty to apply to repeated tokens."""

    top_k: Optional[int] = 40
    """The top-k value to use for sampling."""

    last_n_tokens_size: Optional[int] = 64
    """The number of tokens to look back when applying the repeat_penalty."""

    use_mmap: Optional[bool] = True
    """Whether to keep the model loaded in RAM"""

    streaming: bool = True
    """Whether to stream the results, token by token."""

    verbose: bool = False
    """Whether to print the results."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that llama-cpp-python library is installed."""
        model_path = values["model_path"]
        model_param_names = [
            "lora_path",
            "lora_base",
            "n_ctx",
            "n_parts",
            "seed",
            "f16_kv",
            "logits_all",
            "vocab_only",
            "use_mlock",
            "n_threads",
            "n_batch",
            "use_mmap",
            "last_n_tokens_size",
            # "streaming",
            "verbose",
        ]
        model_params = {k: values[k] for k in model_param_names}
        # For backwards compatibility, only include if non-null.
        if values["n_gpu_layers"] is not None:
            model_params["n_gpu_layers"] = values["n_gpu_layers"]

        try:
            from llama_cpp import Llama

            values["client"] = Llama(model_path, **model_params)
        except ImportError:
            raise ModuleNotFoundError(
                "Could not import llama-cpp-python library. "
                "Please install the llama-cpp-python library to "
                "use this embedding model: pip install llama-cpp-python"
            )
        except Exception as e:
            raise ValueError(
                f"Could not load Llama model from path: {model_path}. "
                f"Received error {e}"
            )

        return values

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "llama.cpp"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling llama_cpp."""
        return {
            # "suffix": self.suffix,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            # "logprobs": self.logprobs,
            # "echo": self.echo,
            "stop_sequences": self.stop,  # key here is convention among LLM classes
            "repeat_penalty": self.repeat_penalty,
            "top_k": self.top_k,
        }

    def _get_parameters(
        self, stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Performs sanity check, preparing paramaters in format needed by llama_cpp.

        Args:
            stop (Optional[List[str]]): List of stop sequences for llama_cpp.

        Returns:
            Dictionary containing the combined parameters.
        """

        # Raise error if stop sequences are in both input and default params
        if self.stop and stop is not None:
            raise ValueError(
                "`stop` found in both the input and default params."
            )

        params = self._default_params

        # llama_cpp expects the "stop" key not this, so we remove it:
        params.pop("stop_sequences")

        # then sets it as configured, or default to an empty list:
        params["stop"] = self.stop or stop or []

        return params

    def _messageConverter(
        self, messages: List[BaseMessage]
    ) -> List[ChatCompletionMessage]:
        chat_messages: List[ChatCompletionMessage] = []
        for message in messages:
            role = "assistant"
            if message.type == "human":
                role = "user"
            elif message.type == "system":
                role = "system"
            else:
                role = "assistant"
            chat_message = ChatCompletionMessage(
                content=message.content,
                role=role,
            )
            chat_messages.append(chat_message)
        return chat_messages

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> Generator[Dict, None, None]:
        params = self._get_parameters(stop)
        result = self.client.create_chat_completion(
            self._messageConverter(messages),
            stream=True,
            **params,
        )
        for chunk in result:
            # print(chunk)
            token = chunk["choices"][0]["delta"].get("content", "")
            log_probs = chunk["choices"][0].get("logprobs", None)
            if run_manager:
                run_manager.on_llm_new_token(
                    token=token, verbose=self.verbose, log_probs=log_probs
                )
            yield chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        if self.streaming:
            output_str = ""
            for chunk in self._stream(
                messages, stop=stop, run_manager=run_manager
            ):
                output_str += chunk["choices"][0]["delta"].get("content", "")
        else:
            params = self._get_parameters(stop)
            result = self.client.create_chat_completion(
                self._messageConverter(messages), **params
            )
            output_str = result["choices"][0]["delta"].get("content", "")

        message = AIMessage(content=output_str)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        if self.streaming:
            output_str = ""
            for chunk in self._stream(
                messages, stop=stop, run_manager=run_manager
            ):
                output_str += chunk["choices"][0]["delta"].get("content", "")
        else:
            params = self._get_parameters(stop)
            result = self.client.create_chat_completion(
                self._messageConverter(messages), **params
            )
            output_str = result["choices"][0]["delta"].get("content", "")

        message = AIMessage(content=output_str)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
