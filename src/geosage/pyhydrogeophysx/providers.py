"""Use the host's provider clients for the existing GeoSAGE report agents."""

from types import SimpleNamespace


class StudioLLM:
    """The small chat interface used by GeoSAGE, backed by a host BaseAgent.

    No key means no network calls, even if the process has provider keys in its
    environment. Session credentials never enter serialized configuration.
    """

    def __init__(self, settings):
        self.api_key = settings.get("api_key")
        self.model = settings.get("model")
        self.provider = settings.get("llm_provider", "openai")
        self._agent = None
        self.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=self._create))
        )

    def query(self, prompt, system_message=None, temperature=0.0, max_tokens=8000, on_text=None):
        if not self.api_key:
            raise RuntimeError("Configure a session API key in the Studio assistant panel.")
        if self._agent is None:
            from PyHydroGeophysX.agents.base_agent import BaseAgent

            class ReportBackend(BaseAgent):
                def execute(self, input_data):
                    return {}

            self._agent = ReportBackend(
                "geosage", api_key=self.api_key, model=self.model, llm_provider=self.provider
            )
        try:
            response = self._agent.query_llm(
                prompt,
                system_message=system_message,
                temperature=temperature,
                max_tokens=max_tokens,
                on_text=on_text,
            )
        except Exception as exc:
            detail = str(exc).replace(str(self.api_key), "[REDACTED]")
            raise RuntimeError(f"Studio provider failed: {detail}") from None
        if not isinstance(response, str) or not response.strip():
            raise RuntimeError("Studio provider returned an empty text response.")
        return response

    def _create(self, *, messages, temperature=0.0, **kwargs):
        system = "\n\n".join(str(m["content"]) for m in messages if m["role"] == "system")
        prompt = "\n\n".join(str(m["content"]) for m in messages if m["role"] != "system")
        content = self.query(prompt, system, temperature, kwargs.get("max_tokens", 8000))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    def chat_json(self, system_prompt, user_prompt, temperature=0.0):
        from geosage.multi_agent_runner import LLMClient

        text = self.query(user_prompt, system_prompt + "\nReturn a JSON object only.", temperature)
        result = LLMClient._safe_json_loads(text)
        if not isinstance(result, dict):
            raise ValueError("Studio provider must return a JSON object.")
        return result
