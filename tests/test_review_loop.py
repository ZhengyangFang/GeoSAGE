from __future__ import annotations

import json
from pathlib import Path

from test_interpret_existing import _config, _make_source
from geosage.multi_agent_runner import MultiAgentOrchestrator


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **_: object):
        self.calls += 1
        class Message:
            content = "# revised report"

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]

        return Response()


class _FakeLLM:
    model = "fake"

    def __init__(self, decision: str = "REVISE_REPORT") -> None:
        self.completions = _FakeCompletions()
        self.client = type("Client", (), {"chat": type("Chat", (), {"completions": self.completions})()})()
        self.decision = decision
        self.review_prompts: list[str] = []

    def chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.0):
        if "Audit this draft" in user_prompt:
            self.review_prompts.append(user_prompt)
            if self.decision != "REVISE_REPORT":
                return {
                    "decision": self.decision,
                    "summary": "The structured evidence is sufficient." if self.decision == "ACCEPT" else "The evidence is insufficient.",
                    "issues": [],
                }
            return {
                "decision": "REVISE_REPORT",
                "summary": "Correct one deliberate issue.",
                "issues": [
                    {
                        "claim_id": "C001",
                        "severity": "major",
                        "category": "coordinate",
                        "problem": "Layer index was used as depth.",
                        "evidence": "E_MESH_GEOMETRY",
                        "required_action": "Use the physical z coordinate.",
                    }
                ],
            }
        return {}


def test_review_revision_is_one_round(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    cfg = _config(source, tmp_path / "review_out")
    cfg["run"].update({"write_reports": True, "review_enabled": True, "max_review_rounds": 9})
    fake_llm = _FakeLLM()
    result = MultiAgentOrchestrator(llm=fake_llm).run_from_config(cfg, user_request="review")
    assert result["review"]["decision"] == "REVISE_REPORT"
    assert Path(result["draft_report_path"]).read_text(encoding="utf-8").strip() == "# revised report"
    review_path = Path(result["workflow_result"]["interpretation_output_dir"]) / "reviews" / "review_round_1.json"
    assert json.loads(review_path.read_text(encoding="utf-8"))["decision"] == "REVISE_REPORT"
    revision = json.loads(
        (review_path.parent / "revision_summary.json").read_text(encoding="utf-8")
    )
    assert revision["rounds"] == 1
    assert result["review"]["issues"][0]["category"] == "coordinate"
    assert "Model z is ELEVATION" in fake_llm.review_prompts[0]
    assert "E_COORDINATE_REFERENCE" in fake_llm.review_prompts[0]
    assert "E_TARGET_DEPTH_AUDIT" in fake_llm.review_prompts[0]
    assert "k is an array index only" in fake_llm.review_prompts[0]
    trace = json.loads(
        (Path(result["workflow_result"]["interpretation_output_dir"]) / "agent_trace.json").read_text(encoding="utf-8")
    )
    assert any(item["agent"] == "ReviewAgent" for item in trace["interactions"])


def test_review_accept_keeps_draft_as_final(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    cfg = _config(source, tmp_path / "accept_out")
    cfg["run"].update({"write_reports": True, "review_enabled": True})
    result = MultiAgentOrchestrator(llm=_FakeLLM("ACCEPT")).run_from_config(cfg, user_request="accept")
    assert result["review"]["decision"] == "ACCEPT"
    assert Path(result["draft_report_path"]).read_text(encoding="utf-8") == Path(result["report_path"]).read_text(encoding="utf-8")


def test_review_insufficient_evidence_preserves_draft(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    cfg = _config(source, tmp_path / "insufficient_out")
    cfg["run"].update({"write_reports": True, "review_enabled": True})
    result = MultiAgentOrchestrator(llm=_FakeLLM("INSUFFICIENT_EVIDENCE")).run_from_config(
        cfg,
        user_request="insufficient",
    )
    assert result["review"]["decision"] == "INSUFFICIENT_EVIDENCE"
    assert Path(result["draft_report_path"]).is_file()
    assert result["report"].startswith("# Insufficient evidence")
