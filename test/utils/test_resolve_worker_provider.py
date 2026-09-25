"""Fail-closed provider resolution for explicitly named worker profiles.

handoff/assign always name the child profile explicitly. Before this
hardening, a profile that failed to LOAD (missing file, unreadable, malformed
frontmatter) was silently resolved to the supervisor's provider by
``resolve_provider()``'s legacy fallback -- observed live as
``handoff("kimi-developer")`` launching a ``codex --yolo`` worker after the
kimi_cli profile lookup failed. These tests pin the fail-closed contract of
``_resolve_worker_provider``: an explicitly named profile either loads and
resolves, or the handoff/assign fails with an explicit error and ZERO side
effects -- no run-step POST, no terminal creation, no worker process.
"""

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest

from cli_agent_orchestrator.constants import PROVIDERS
from cli_agent_orchestrator.models.agent_profile import AgentProfile
from cli_agent_orchestrator.utils.orchestration import (
    _assign_impl,
    _handoff_impl,
    _resolve_worker_provider,
)

ORCH = "cli_agent_orchestrator.utils.orchestration"
MISSING_PROFILE = "definitely-does-not-exist"


def _supervisor_metadata(provider="codex"):
    """A mocked GET /terminals/{id} response for a supervisor terminal.

    Defaults to a CODEX supervisor on purpose: the original bug was a missing
    kimi_cli profile silently inheriting exactly this provider.
    """
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "provider": provider,
        "session_name": "cao-sup",
        "allowed_tools": None,
    }
    resp.raise_for_status.return_value = None
    return resp


def _ok_run_step_response(terminal_id="dev-term", last_message="task done"):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "terminal_id": terminal_id,
        "last_message": last_message,
        "status": "completed",
    }
    resp.raise_for_status.return_value = None
    return resp


def _ok_create_response(terminal_id="worker-1"):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"id": terminal_id}
    resp.raise_for_status.return_value = None
    return resp


class TestResolveWorkerProvider:
    """Unit tests for the fail-closed resolver itself."""

    @patch(f"{ORCH}.load_agent_profile")
    def test_valid_custom_profile_overrides_parent_provider(self, mock_load):
        """A valid kimi_cli profile resolves to kimi_cli even when the
        supervisor runs codex -- the exact inversion of the original bug."""
        mock_load.return_value = AgentProfile(
            name="kimi-developer", description="Kimi developer", provider="kimi_cli"
        )

        result = _resolve_worker_provider("kimi-developer", fallback_provider="codex")

        assert result == "kimi_cli"
        mock_load.assert_called_once_with("kimi-developer")

    @patch(f"{ORCH}.load_agent_profile")
    def test_profile_without_provider_inherits_fallback(self, mock_load):
        """Documented inheritance: a loaded profile that OMITS the provider key
        still inherits the caller's provider (fallback is preserved for this
        case only)."""
        mock_load.return_value = AgentProfile(name="reviewer", description="Reviewer")

        assert _resolve_worker_provider("reviewer", fallback_provider="codex") == "codex"

    @patch(f"{ORCH}.load_agent_profile")
    def test_profile_with_empty_provider_inherits_fallback(self, mock_load):
        mock_load.return_value = AgentProfile(name="reviewer", description="Reviewer", provider="")

        assert _resolve_worker_provider("reviewer", fallback_provider="codex") == "codex"

    @patch(f"{ORCH}.load_agent_profile")
    def test_missing_profile_fails_closed(self, mock_load):
        mock_load.side_effect = FileNotFoundError(f"Agent profile not found: {MISSING_PROFILE}")

        with pytest.raises(ValueError) as exc_info:
            _resolve_worker_provider(MISSING_PROFILE, fallback_provider="codex")

        message = str(exc_info.value)
        assert MISSING_PROFILE in message
        assert "could not be loaded" in message
        assert "no worker was launched" in message
        # The parent provider must not leak into the failure as a suggestion.
        mock_load.assert_called_once_with(MISSING_PROFILE)

    @patch(f"{ORCH}.load_agent_profile")
    def test_malformed_profile_fails_closed(self, mock_load):
        """Unparseable frontmatter surfaces as RuntimeError from the loader."""
        mock_load.side_effect = RuntimeError(
            f"Failed to load agent profile '{MISSING_PROFILE}': bad yaml"
        )

        with pytest.raises(ValueError, match="could not be loaded"):
            _resolve_worker_provider(MISSING_PROFILE, fallback_provider="codex")

    @patch(f"{ORCH}.load_agent_profile")
    def test_validation_error_fails_closed(self, mock_load):
        """Schema/validation failures (ValueError from the loader, incl.
        pydantic ValidationError and agent-name rejection) fail closed too."""
        mock_load.side_effect = ValueError("Invalid agent name")

        with pytest.raises(ValueError, match="could not be loaded"):
            _resolve_worker_provider(MISSING_PROFILE, fallback_provider="codex")

    @patch(f"{ORCH}.load_agent_profile")
    def test_invalid_provider_value_fails_closed(self, mock_load):
        """A loaded profile whose provider is not a known provider is a
        misconfiguration, not a legitimate omission -- inheriting there would
        be the same silent provider swap with a typo'd key."""
        mock_load.return_value = AgentProfile(
            name="typo-agent", description="typo", provider="codexx"
        )

        with pytest.raises(ValueError, match="invalid provider 'codexx'"):
            _resolve_worker_provider("typo-agent", fallback_provider="kimi_cli")

    @patch(f"{ORCH}.load_agent_profile")
    def test_all_valid_provider_types_accepted(self, mock_load):
        for provider_value in PROVIDERS:
            mock_load.return_value = AgentProfile(
                name="agent", description="test", provider=provider_value
            )
            assert _resolve_worker_provider("agent", fallback_provider="kiro_cli") == provider_value

    def test_builtin_profile_resolves_via_real_store(self, monkeypatch, tmp_path):
        """A real built-in profile (no mocking of the loader) still resolves:
        the built-in ``developer`` declares no provider key, so the documented
        inheritance applies and the fallback is returned."""
        monkeypatch.setattr(
            "cli_agent_orchestrator.utils.agent_profiles.LOCAL_AGENT_STORE_DIR", tmp_path
        )
        monkeypatch.setattr(
            "cli_agent_orchestrator.services.settings_service.get_agent_dirs", lambda: {}
        )
        monkeypatch.setattr(
            "cli_agent_orchestrator.services.settings_service.get_disabled_agent_dirs",
            lambda: [],
        )
        monkeypatch.setattr(
            "cli_agent_orchestrator.services.settings_service.get_extra_agent_dirs",
            lambda: [],
        )

        assert _resolve_worker_provider("developer", fallback_provider="codex") == "codex"


class TestHandoffFailClosed:
    """handoff must fail closed and issue ZERO run-step POSTs when the
    explicit child profile cannot be loaded."""

    @patch(f"{ORCH}.load_agent_profile")
    def test_missing_profile_inside_terminal(self, mock_load):
        mock_load.side_effect = FileNotFoundError(f"Agent profile not found: {MISSING_PROFILE}")

        with patch.dict(os.environ, {"CAO_TERMINAL_ID": "a1b2c3d4"}):
            with patch(f"{ORCH}.requests") as mock_requests:
                mock_requests.get.return_value = _supervisor_metadata(provider="codex")
                result = asyncio.run(_handoff_impl(MISSING_PROFILE, "do the thing"))

        assert result.success is False
        assert result.terminal_id is None
        assert MISSING_PROFILE in result.message
        assert "could not be loaded" in result.message
        # Zero side effects: no run-step, no terminal creation, nothing.
        mock_requests.post.assert_not_called()

    @patch(f"{ORCH}.load_agent_profile")
    def test_missing_profile_outside_terminal(self, mock_load):
        """Without CAO_TERMINAL_ID the resolver runs before ANY HTTP call."""
        mock_load.side_effect = FileNotFoundError(f"Agent profile not found: {MISSING_PROFILE}")

        with patch.dict(os.environ, {}, clear=True):
            with patch(f"{ORCH}.requests") as mock_requests:
                result = asyncio.run(_handoff_impl(MISSING_PROFILE, "do the thing"))

        assert result.success is False
        assert MISSING_PROFILE in result.message
        mock_requests.post.assert_not_called()
        mock_requests.get.assert_not_called()

    @pytest.mark.parametrize(
        "load_error",
        [
            RuntimeError("Failed to load agent profile: bad yaml"),
            ValueError("frontmatter validation failed"),
        ],
        ids=["runtime-error", "validation-error"],
    )
    @patch(f"{ORCH}.load_agent_profile")
    def test_malformed_profile_fails_closed(self, mock_load, load_error):
        mock_load.side_effect = load_error

        with patch.dict(os.environ, {"CAO_TERMINAL_ID": "a1b2c3d4"}):
            with patch(f"{ORCH}.requests") as mock_requests:
                mock_requests.get.return_value = _supervisor_metadata(provider="codex")
                result = asyncio.run(_handoff_impl(MISSING_PROFILE, "do the thing"))

        assert result.success is False
        assert result.terminal_id is None
        assert "could not be loaded" in result.message
        mock_requests.post.assert_not_called()

    @patch(f"{ORCH}._get_cleanup_nudge", return_value="")
    @patch(f"{ORCH}._resolve_child_allowed_tools", return_value=None)
    @patch(f"{ORCH}.load_agent_profile")
    def test_valid_kimi_profile_resolves_kimi_cli_not_supervisor_codex(
        self, mock_load, _tools, _nudge
    ):
        """The original failure inverted: supervisor is codex, child profile is
        kimi_cli, and the run-step payload must carry kimi_cli -- never the
        inherited codex."""
        mock_load.return_value = AgentProfile(
            name="kimi-developer", description="Kimi developer", provider="kimi_cli"
        )

        with patch.dict(os.environ, {"CAO_TERMINAL_ID": "a1b2c3d4"}):
            with patch(f"{ORCH}.requests") as mock_requests:
                mock_requests.get.return_value = _supervisor_metadata(provider="codex")
                mock_requests.post.return_value = _ok_run_step_response()
                mock_requests.Timeout = Exception
                result = asyncio.run(_handoff_impl("kimi-developer", "do the thing"))

        assert result.success is True
        payload = mock_requests.post.call_args[1]["json"]
        assert payload["provider"] == "kimi_cli"
        assert payload["agent"] == "kimi-developer"


class TestAssignFailClosed:
    """assign must fail closed and create ZERO terminals when the explicit
    child profile cannot be loaded."""

    @patch(f"{ORCH}.load_agent_profile")
    def test_missing_profile_fails_closed(self, mock_load):
        mock_load.side_effect = FileNotFoundError(f"Agent profile not found: {MISSING_PROFILE}")

        with patch.dict(os.environ, {"CAO_TERMINAL_ID": "a1b2c3d4"}):
            with patch(f"{ORCH}.requests") as mock_requests:
                mock_requests.get.return_value = _supervisor_metadata(provider="codex")
                result = _assign_impl(MISSING_PROFILE, "do the thing")

        assert result["success"] is False
        assert result["terminal_id"] is None
        assert MISSING_PROFILE in result["message"]
        assert "could not be loaded" in result["message"]
        # Zero side effects: no terminal-creation POST.
        mock_requests.post.assert_not_called()

    @pytest.mark.parametrize(
        "load_error",
        [
            RuntimeError("Failed to load agent profile: bad yaml"),
            ValueError("frontmatter validation failed"),
        ],
        ids=["runtime-error", "validation-error"],
    )
    @patch(f"{ORCH}.load_agent_profile")
    def test_malformed_profile_fails_closed(self, mock_load, load_error):
        mock_load.side_effect = load_error

        with patch.dict(os.environ, {"CAO_TERMINAL_ID": "a1b2c3d4"}):
            with patch(f"{ORCH}.requests") as mock_requests:
                mock_requests.get.return_value = _supervisor_metadata(provider="codex")
                result = _assign_impl(MISSING_PROFILE, "do the thing")

        assert result["success"] is False
        assert result["terminal_id"] is None
        assert "could not be loaded" in result["message"]
        mock_requests.post.assert_not_called()

    @patch(f"{ORCH}._resolve_child_allowed_tools", return_value=None)
    @patch(f"{ORCH}.load_agent_profile")
    def test_valid_kimi_profile_assigned_with_kimi_cli(self, mock_load, _tools):
        mock_load.return_value = AgentProfile(
            name="kimi-developer", description="Kimi developer", provider="kimi_cli"
        )

        with patch.dict(os.environ, {"CAO_TERMINAL_ID": "a1b2c3d4"}):
            with patch(f"{ORCH}.requests") as mock_requests:
                mock_requests.get.return_value = _supervisor_metadata(provider="codex")
                mock_requests.post.return_value = _ok_create_response()
                result = _assign_impl("kimi-developer", "do the thing")

        assert result["success"] is True
        assert result["terminal_id"] == "worker-1"
        params = mock_requests.post.call_args[1]["params"]
        assert params["provider"] == "kimi_cli"
        assert params["agent_profile"] == "kimi-developer"
