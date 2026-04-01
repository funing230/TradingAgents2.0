"""
Unit tests for LLMPool — multi-model role-based assignment with mode and failover.
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from dataclasses import dataclass

from tradingagents.llm_clients.pool import LLMPool, ResilientLLM


# ===================================================================
# Test configs
# ===================================================================

POOL_CONFIG = {
    "llm_pool": {
        "model-a": {
            "provider": "openai",
            "model": "test-model-a",
            "base_url": "http://localhost:1111/v1",
            "api_key": "test-key-a",
            "modes": {
                "chat": {},
                "deepthink": {"reasoning_effort": "high"},
            },
        },
        "model-b": {
            "provider": "openai",
            "model": "test-model-b",
            "base_url": "http://localhost:2222/v1",
            "api_key": "test-key-b",
            "modes": {
                "chat": {},
                "deepthink": {"reasoning_effort": "high"},
            },
        },
        "model-c": {
            "provider": "openai",
            "model": "test-model-c",
            "base_url": "http://localhost:3333/v1",
            "api_key": "test-key-c",
            "modes": {
                "chat": {},
                "deepthink": {"thinking_level": "high"},
            },
        },
    },
    "llm_roles": {
        "market_analyst":       {"model": "model-a", "mode": "chat"},
        "news_analyst":         {"model": "model-a", "mode": "chat"},
        "research_manager":     {"model": "model-b", "mode": "deepthink",
                                 "fallback": [{"model": "model-a", "mode": "deepthink"}]},
        "portfolio_manager":    {"model": "model-b", "mode": "deepthink",
                                 "fallback": [{"model": "model-a", "mode": "deepthink"}]},
        "aggressive_debater":   {"model": "model-c", "mode": "chat"},
        "conservative_debater": {"model": "model-c", "mode": "chat"},
        "neutral_debater":      {"model": "model-c", "mode": "chat"},
        "signal_processor":     {"model": "model-a", "mode": "chat"},
        "reflector":            {"model": "model-b", "mode": "deepthink"},
    },
    # Legacy fallback
    "llm_provider": "openai",
    "deep_think_llm": "legacy-deep",
    "quick_think_llm": "legacy-quick",
    "backend_url": "http://localhost:9999/v1",
}

# Config with legacy string-format roles (backward compat)
LEGACY_ROLES_CONFIG = {
    "llm_pool": {
        "model-a": {
            "provider": "openai",
            "model": "test-model-a",
            "base_url": "http://localhost:1111/v1",
            "api_key": "test-key-a",
        },
    },
    "llm_roles": {
        "market_analyst": "model-a",  # legacy string format
    },
    "llm_provider": "openai",
    "deep_think_llm": "legacy-deep",
    "quick_think_llm": "legacy-quick",
    "backend_url": "http://localhost:9999/v1",
}

LEGACY_CONFIG = {
    # No llm_pool or llm_roles
    "llm_provider": "openai",
    "deep_think_llm": "legacy-deep",
    "quick_think_llm": "legacy-quick",
    "backend_url": "http://localhost:9999/v1",
}


# ===================================================================
# Tests — Role mapping with modes
# ===================================================================


class TestLLMPoolRoleMapping:
    """Test role → model+mode resolution."""

    @patch("tradingagents.llm_clients.pool.create_llm_client")
    def test_chat_role_no_reasoning_kwargs(self, mock_create):
        mock_client = MagicMock()
        mock_client.get_llm.return_value = MagicMock(name="llm-instance")
        mock_create.return_value = mock_client

        pool = LLMPool(POOL_CONFIG)
        pool.get_llm("market_analyst")

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["model"] == "test-model-a"
        assert "reasoning_effort" not in call_kwargs

    @patch("tradingagents.llm_clients.pool.create_llm_client")
    def test_deepthink_role_has_reasoning_kwargs(self, mock_create):
        mock_client = MagicMock()
        mock_client.get_llm.return_value = MagicMock(name="llm-instance")
        mock_create.return_value = mock_client

        pool = LLMPool(POOL_CONFIG)
        pool.get_llm("research_manager")

        # Primary is model-b with deepthink
        first_call = mock_create.call_args_list[0][1]
        assert first_call["model"] == "test-model-b"
        assert first_call["reasoning_effort"] == "high"

    @patch("tradingagents.llm_clients.pool.create_llm_client")
    def test_different_roles_get_different_models(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        pool = LLMPool(POOL_CONFIG)

        pool.get_llm("market_analyst")
        call1_model = mock_create.call_args_list[-1][1]["model"]

        pool.get_llm("aggressive_debater")
        call2_model = mock_create.call_args_list[-1][1]["model"]

        assert call1_model == "test-model-a"
        assert call2_model == "test-model-c"

    @patch("tradingagents.llm_clients.pool.create_llm_client")
    def test_legacy_string_role_treated_as_chat(self, mock_create):
        mock_client = MagicMock()
        mock_client.get_llm.return_value = MagicMock()
        mock_create.return_value = mock_client

        pool = LLMPool(LEGACY_ROLES_CONFIG)
        pool.get_llm("market_analyst")

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["model"] == "test-model-a"
        assert "reasoning_effort" not in call_kwargs


# ===================================================================
# Tests — Caching (mode-aware)
# ===================================================================


class TestLLMPoolCaching:
    """Test singleton caching with mode awareness."""

    @patch("tradingagents.llm_clients.pool.create_llm_client")
    def test_same_model_same_mode_cached(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        pool = LLMPool(POOL_CONFIG)

        llm1 = pool.get_llm("market_analyst")   # model-a:chat
        llm2 = pool.get_llm("news_analyst")     # also model-a:chat

        model_a_calls = [
            c for c in mock_create.call_args_list
            if c[1].get("model") == "test-model-a"
        ]
        assert len(model_a_calls) == 1

    @patch("tradingagents.llm_clients.pool.create_llm_client")
    def test_same_model_different_mode_separate(self, mock_create):
        mock_client_chat = MagicMock()
        mock_client_chat.get_llm.return_value = "llm-chat"
        mock_client_deep = MagicMock()
        mock_client_deep.get_llm.return_value = "llm-deep"
        mock_create.side_effect = [mock_client_chat, mock_client_deep]

        pool = LLMPool(POOL_CONFIG)

        llm_chat = pool.get_llm_by_key("model-a", mode="chat")
        llm_deep = pool.get_llm_by_key("model-a", mode="deepthink")

        assert llm_chat != llm_deep
        assert mock_create.call_count == 2


# ===================================================================
# Tests — Failover
# ===================================================================


class TestLLMPoolFailover:
    """Test fallback chain and ResilientLLM."""

    @patch("tradingagents.llm_clients.pool.create_llm_client")
    def test_role_with_fallback_returns_resilient(self, mock_create):
        mock_client = MagicMock()
        mock_client.get_llm.return_value = MagicMock()
        mock_create.return_value = mock_client

        pool = LLMPool(POOL_CONFIG)
        llm = pool.get_llm("research_manager")

        # Should be wrapped in ResilientLLM because it has fallback
        assert isinstance(llm, ResilientLLM)

    @patch("tradingagents.llm_clients.pool.create_llm_client")
    def test_role_without_fallback_returns_raw(self, mock_create):
        mock_client = MagicMock()
        mock_client.get_llm.return_value = MagicMock()
        mock_create.return_value = mock_client

        pool = LLMPool(POOL_CONFIG)
        llm = pool.get_llm("market_analyst")

        # No fallback → raw LLM, not wrapped
        assert not isinstance(llm, ResilientLLM)

    @patch("tradingagents.llm_clients.pool.create_llm_client")
    def test_disabled_primary_uses_fallback(self, mock_create):
        mock_client = MagicMock()
        mock_client.get_llm.return_value = MagicMock()
        mock_create.return_value = mock_client

        pool = LLMPool(POOL_CONFIG)
        pool._disabled_models.add("model-b")

        llm = pool.get_llm("research_manager")
        # Primary model-b disabled, should still work via fallback model-a
        assert llm is not None


class TestResilientLLM:
    """Test ResilientLLM retry and failover logic."""

    def test_primary_success(self):
        primary = MagicMock()
        primary.invoke.return_value = "ok"
        fallback = MagicMock()

        llm = ResilientLLM(primary, [fallback])
        result = llm.invoke("test")

        assert result == "ok"
        primary.invoke.assert_called_once()
        fallback.invoke.assert_not_called()

    def test_primary_transient_error_retries(self):
        primary = MagicMock()
        primary.invoke.side_effect = [
            ConnectionError("timeout connecting"),
            "ok",
        ]

        llm = ResilientLLM(primary, [], max_retries=2)
        result = llm.invoke("test")

        assert result == "ok"
        assert primary.invoke.call_count == 2

    def test_primary_fails_falls_to_fallback(self):
        primary = MagicMock()
        primary.invoke.side_effect = ConnectionError("timeout")
        fallback = MagicMock()
        fallback.invoke.return_value = "fallback-ok"

        llm = ResilientLLM(primary, [fallback], max_retries=1)
        result = llm.invoke("test")

        assert result == "fallback-ok"

    def test_non_transient_skips_retries(self):
        primary = MagicMock()
        primary.invoke.side_effect = ValueError("invalid input")
        fallback = MagicMock()
        fallback.invoke.return_value = "fallback-ok"

        llm = ResilientLLM(primary, [fallback], max_retries=3)
        result = llm.invoke("test")

        # Non-transient: should NOT retry, go straight to fallback
        assert primary.invoke.call_count == 1
        assert result == "fallback-ok"

    def test_all_fail_raises(self):
        primary = MagicMock()
        primary.invoke.side_effect = ConnectionError("timeout")
        fallback = MagicMock()
        fallback.invoke.side_effect = ConnectionError("timeout")

        llm = ResilientLLM(primary, [fallback], max_retries=1)
        with pytest.raises(ConnectionError):
            llm.invoke("test")


# ===================================================================
# Tests — Probe integration
# ===================================================================


class TestLLMPoolProbeIntegration:
    """Test apply_probe_results behavior."""

    @dataclass
    class FakeProbeResult:
        available: bool = True
        deepthink_ok: bool = True

    @patch("tradingagents.llm_clients.pool.create_llm_client")
    def test_unavailable_model_disabled(self, mock_create):
        mock_client = MagicMock()
        mock_client.get_llm.return_value = MagicMock()
        mock_create.return_value = mock_client

        pool = LLMPool(POOL_CONFIG)
        pool.apply_probe_results({
            "model-a": self.FakeProbeResult(available=False),
            "model-b": self.FakeProbeResult(available=True),
        })

        assert "model-a" in pool._disabled_models
        assert "model-b" not in pool._disabled_models

    @patch("tradingagents.llm_clients.pool.create_llm_client")
    def test_no_deepthink_downgrades_to_chat(self, mock_create):
        mock_client = MagicMock()
        mock_client.get_llm.return_value = MagicMock()
        mock_create.return_value = mock_client

        pool = LLMPool(POOL_CONFIG)
        pool.apply_probe_results({
            "model-b": self.FakeProbeResult(available=True, deepthink_ok=False),
        })

        # Request deepthink, but model-b has no deepthink → downgrade to chat
        pool.get_llm_by_key("model-b", mode="deepthink")

        call_kwargs = mock_create.call_args[1]
        # Should NOT have reasoning_effort since it was downgraded to chat
        assert "reasoning_effort" not in call_kwargs


# ===================================================================
# Tests — Legacy fallback
# ===================================================================


class TestLLMPoolLegacyFallback:
    """Test backward compatibility when llm_roles is not configured."""

    @patch("tradingagents.llm_clients.pool.create_llm_client")
    def test_no_roles_uses_legacy(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        pool = LLMPool(LEGACY_CONFIG)

        pool.get_llm("market_analyst")
        assert mock_create.call_args_list[-1][1]["model"] == "legacy-quick"

    @patch("tradingagents.llm_clients.pool.create_llm_client")
    def test_decision_roles_use_deep(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        pool = LLMPool(LEGACY_CONFIG)

        pool.get_llm("research_manager")
        assert mock_create.call_args_list[-1][1]["model"] == "legacy-deep"


# ===================================================================
# Tests — Direct access and metadata
# ===================================================================


class TestLLMPoolDirectAccess:
    """Test get_llm_by_key for direct model access."""

    @patch("tradingagents.llm_clients.pool.create_llm_client")
    def test_get_by_key_with_mode(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        pool = LLMPool(POOL_CONFIG)
        pool.get_llm_by_key("model-c", mode="deepthink")

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["model"] == "test-model-c"
        assert call_kwargs["thinking_level"] == "high"

    def test_invalid_key_raises(self):
        pool = LLMPool(POOL_CONFIG)
        with pytest.raises(ValueError, match="not found in llm_pool"):
            pool.get_llm_by_key("nonexistent")


class TestLLMPoolMetadata:
    """Test metadata methods."""

    def test_get_all_keys(self):
        pool = LLMPool(POOL_CONFIG)
        keys = pool.get_all_keys()
        assert set(keys) == {"model-a", "model-b", "model-c"}

    def test_get_role_mapping(self):
        pool = LLMPool(POOL_CONFIG)
        mapping = pool.get_role_mapping()
        assert mapping["market_analyst"]["model"] == "model-a"
        assert mapping["research_manager"]["mode"] == "deepthink"

    def test_empty_pool(self):
        pool = LLMPool({"llm_pool": {}})
        assert pool.get_all_keys() == []
