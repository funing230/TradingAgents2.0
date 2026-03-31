"""
Unit tests for LLMPool — multi-model role-based assignment.
"""

import pytest
from unittest.mock import patch, MagicMock

from tradingagents.llm_clients.pool import LLMPool


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
        },
        "model-b": {
            "provider": "openai",
            "model": "test-model-b",
            "base_url": "http://localhost:2222/v1",
            "api_key": "test-key-b",
        },
        "model-c": {
            "provider": "openai",
            "model": "test-model-c",
            "base_url": "http://localhost:3333/v1",
            "api_key": "test-key-c",
        },
    },
    "llm_roles": {
        "market_analyst": "model-a",
        "news_analyst": "model-a",
        "research_manager": "model-b",
        "portfolio_manager": "model-b",
        "aggressive_debater": "model-c",
        "conservative_debater": "model-c",
        "neutral_debater": "model-c",
    },
    # Legacy fallback
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
# Tests
# ===================================================================


class TestLLMPoolRoleMapping:
    """Test role → model resolution."""

    @patch("tradingagents.llm_clients.pool.create_llm_client")
    def test_role_resolves_to_correct_model(self, mock_create):
        mock_client = MagicMock()
        mock_client.get_llm.return_value = MagicMock(name="llm-instance")
        mock_create.return_value = mock_client

        pool = LLMPool(POOL_CONFIG)

        pool.get_llm("market_analyst")
        mock_create.assert_called_with(
            provider="openai",
            model="test-model-a",
            base_url="http://localhost:1111/v1",
            api_key="test-key-a",
        )

    @patch("tradingagents.llm_clients.pool.create_llm_client")
    def test_different_roles_get_different_models(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        pool = LLMPool(POOL_CONFIG)

        pool.get_llm("market_analyst")
        call1_model = mock_create.call_args_list[-1][1]["model"]

        pool.get_llm("research_manager")
        call2_model = mock_create.call_args_list[-1][1]["model"]

        assert call1_model == "test-model-a"
        assert call2_model == "test-model-b"

    @patch("tradingagents.llm_clients.pool.create_llm_client")
    def test_debaters_use_model_c(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        pool = LLMPool(POOL_CONFIG)

        pool.get_llm("aggressive_debater")
        assert mock_create.call_args_list[-1][1]["model"] == "test-model-c"

        pool.get_llm("conservative_debater")
        # Should use cache, no new create call for same model
        # model-c was already created


class TestLLMPoolCaching:
    """Test singleton caching."""

    @patch("tradingagents.llm_clients.pool.create_llm_client")
    def test_same_model_key_cached(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        pool = LLMPool(POOL_CONFIG)

        llm1 = pool.get_llm("market_analyst")   # model-a
        llm2 = pool.get_llm("news_analyst")     # also model-a

        # Should only create once for model-a
        model_a_calls = [
            c for c in mock_create.call_args_list
            if c[1].get("model") == "test-model-a"
        ]
        assert len(model_a_calls) == 1
        assert llm1 is llm2

    @patch("tradingagents.llm_clients.pool.create_llm_client")
    def test_different_model_keys_separate(self, mock_create):
        mock_client_a = MagicMock()
        mock_client_b = MagicMock()
        mock_client_a.get_llm.return_value = "llm-a"
        mock_client_b.get_llm.return_value = "llm-b"
        mock_create.side_effect = [mock_client_a, mock_client_b]

        pool = LLMPool(POOL_CONFIG)

        llm1 = pool.get_llm("market_analyst")     # model-a
        llm2 = pool.get_llm("research_manager")   # model-b

        assert llm1 != llm2


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

        pool.get_llm("portfolio_manager")
        # Should use cached deep LLM


class TestLLMPoolDirectAccess:
    """Test get_llm_by_key for direct model access."""

    @patch("tradingagents.llm_clients.pool.create_llm_client")
    def test_get_by_key(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        pool = LLMPool(POOL_CONFIG)
        pool.get_llm_by_key("model-c")

        assert mock_create.call_args_list[-1][1]["model"] == "test-model-c"

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
        assert mapping["market_analyst"] == "model-a"
        assert mapping["research_manager"] == "model-b"
        assert mapping["aggressive_debater"] == "model-c"

    def test_empty_pool(self):
        pool = LLMPool({"llm_pool": {}})
        assert pool.get_all_keys() == []
