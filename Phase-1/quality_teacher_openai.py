from __future__ import annotations

from typing import Callable, Literal, Protocol

from quality_teacher_adapters import (
    CompletionRequest,
    ProviderRateLimitCircuitBreaker,
    StructuredResponseFormat,
    TeacherAdapterContractError,
    rate_limit_cooldown_seconds,
)
from quality_teacher_runtime import TeacherGenerationUnavailable


ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh", "max"]


def openai_text_format(response_format: StructuredResponseFormat) -> dict[str, object]:
    if response_format.json_schema is None:
        return {"type": response_format.type}
    return {
        "type": "json_schema",
        "name": response_format.json_schema_name,
        "strict": True,
        "schema": response_format.json_schema,
    }


class ResponsesAPI(Protocol):
    def create(self, **kwargs: object) -> object: ...


class OpenAIClient(Protocol):
    responses: ResponsesAPI


class OpenAIResponsesBackend:
    """OpenAI Responses API transport for bounded Quality-teacher JSON."""

    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: int,
        maximum_transport_retries: int,
        reasoning_effort: ReasoningEffort,
        client_factory: Callable[..., OpenAIClient] | None = None,
    ) -> None:
        if not api_key:
            raise TeacherAdapterContractError(
                teacher_id="openai",
                detail="API key is empty",
            )
        if client_factory is None:
            from openai import OpenAI

            client_factory = OpenAI
        self._client = client_factory(
            api_key=api_key,
            max_retries=maximum_transport_retries,
            timeout=float(timeout_seconds),
        )
        self._reasoning_effort = reasoning_effort
        self._rate_limit = ProviderRateLimitCircuitBreaker()

    def complete(self, request: CompletionRequest) -> str:
        from httpx import TimeoutException
        from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError

        self._rate_limit.raise_if_blocked(request.model_id)
        try:
            response = self._client.responses.create(
                model=request.model_id,
                instructions=request.messages[0].content,
                input=request.messages[1].content,
                max_output_tokens=request.maximum_new_tokens,
                reasoning={"effort": self._reasoning_effort},
                text={
                    "format": openai_text_format(request.response_format),
                    "verbosity": "low",
                },
                store=False,
            )
        except APITimeoutError as error:
            raise TeacherGenerationUnavailable(request.model_id, "read_timeout") from error
        except TimeoutException as error:
            raise TeacherGenerationUnavailable(request.model_id, "read_timeout") from error
        except APIConnectionError as error:
            raise TeacherGenerationUnavailable(request.model_id, "connection_error") from error
        except APIStatusError as error:
            if error.status_code == 429:
                self._rate_limit.trip(rate_limit_cooldown_seconds(error.response.headers))
            raise TeacherGenerationUnavailable(
                request.model_id,
                f"http_status_{error.status_code}",
            ) from error
        except APIError as error:
            raise TeacherGenerationUnavailable(
                request.model_id,
                f"api_error_{type(error).__name__.lower()}",
            ) from error

        content = str(getattr(response, "output_text", "")).strip()
        if not content:
            raise TeacherAdapterContractError(
                teacher_id=request.model_id,
                detail="endpoint returned no textual completion",
            )
        return content
