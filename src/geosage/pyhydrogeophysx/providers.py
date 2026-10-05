"""Use the host's provider clients for the existing GeoSAGE report agents."""

from types import SimpleNamespace

from PyHydroGeophysX.llm import providers as host_providers

# Older hosts still support API-backed workflows. Advertise CLI providers only
# when the installed host actually supplies those adapters.
CLI_PROVIDERS = frozenset(getattr(host_providers, "CLI_PROVIDER_IDS", ()))
STUDIO_PROVIDERS = ("openai", "anthropic", *sorted(CLI_PROVIDERS))


def provider_id(settings):
    """Resolve the desktop and agent spellings without consulting credentials."""
    value = settings.get("provider") or settings.get("llm_provider") or "openai"
    value = host_providers.normalise_provider_id(value)
    if value not in STUDIO_PROVIDERS:
        raise ValueError(f"Unsupported Studio provider: {value}")
    return "claude" if value == "anthropic" else value


def ai_enabled(settings):
    """Explicit local tasks stay offline, even with a saved CLI login."""
    if settings.get("use_ai") is False or settings.get("studio_task") == "inspect":
        return False
    if settings.get("studio_task") == "invert" and settings.get("use_ai") is not True:
        return False
    return bool(settings.get("api_key")) or provider_id(settings) in CLI_PROVIDERS


class StudioLLM:
    """The small chat interface used by GeoSAGE, backed by a host BaseAgent.

    API providers require an explicit session key; CLI providers use the host's
    existing login bridge. Local tasks never call either backend. Session
    credentials never enter serialized configuration.
    """

    def __init__(self, settings):
        self.provider = provider_id(settings)
        self.api_key = None if self.provider in CLI_PROVIDERS else settings.get("api_key")
        self.model = settings.get("model")
        self.enabled = ai_enabled(settings)
        self._agent = None
        self.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=self._create))
        )

    def query(self, prompt, system_message=None, temperature=0.0, max_tokens=8000, on_text=None):
        if not self.enabled:
            raise RuntimeError("AI is disabled for this task or no session API key / CLI provider is configured.")
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
            detail = str(exc)
            if self.api_key:
                detail = detail.replace(str(self.api_key), "[REDACTED]")
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
