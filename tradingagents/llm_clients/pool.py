"""
LLM Pool — Multi-model manager with role-based assignment.

Manages a pool of LLM instances, each with its own provider/endpoint/key.
Agents request LLMs by role name; the pool resolves to the configured model.

Usage:
    pool = LLMPool(config)
    llm = pool.get_llm("market_analyst")      # role-based
    llm = pool.get_llm_by_key("claude-opus")  # direct access

Extensibility:
    - Add new models to config["llm_pool"]
    - Map roles in config["llm_roles"]
    - No code changes needed for new models
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, Optional

from tradingagents.llm_clients import create_llm_client

logger = logging.getLogger(__name__)

# Default role → model fallback mapping
_ROLE_TIER = {
    # Decision makers → deep_think_llm
    "research_manager": "deep",
    "portfolio_manager": "deep",
    # Everyone else → quick_think_llm
    "market_analyst": "quick",
    "news_analyst": "quick",
    "social_analyst": "quick",
    "fundamentals_analyst": "quick",
    "bull_researcher": "quick",
    "bear_researcher": "quick",
    "trader": "quick",
    "aggressive_debater": "quick",
    "conservative_debater": "quick",
    "neutral_debater": "quick",
}


class LLMPool:
    """Multi-model pool with role-based LLM assignment.

    Features:
      - Lazy initialization: LLM instances created on first use
      - Singleton cache: same model_key shares one instance
      - Backward compatible: falls back to deep/quick if llm_roles not set
      - Extensible: add models to llm_pool config without code changes
    """

    def __init__(self, config: Dict[str, Any], callbacks: list = None):
        self.config = config
        self.callbacks = callbacks or []
        self._cache: Dict[str, Any] = {}  # model_key → LLM instance
        self._legacy_cache: Dict[str, Any] = {}  # "deep"/"quick" → LLM

    def get_llm(self, role: str) -> Any:
        """Get LLM instance for a specific agent role.

        Resolution order:
          1. config["llm_roles"][role] → model_key → llm_pool[model_key]
          2. Fallback to legacy deep_think_llm / quick_think_llm
        """
        llm_roles = self.config.get("llm_roles", {})
        model_key = llm_roles.get(role)

        if model_key:
            return self.get_llm_by_key(model_key)

        # Legacy fallback
        tier = _ROLE_TIER.get(role, "quick")
        return self._get_legacy_llm(tier)

    def get_llm_by_key(self, model_key: str) -> Any:
        """Get LLM instance by model pool key (e.g., 'claude-opus')."""
        if model_key in self._cache:
            return self._cache[model_key]

        pool = self.config.get("llm_pool", {})
        if model_key not in pool:
            raise ValueError(
                f"Model '{model_key}' not found in llm_pool. "
                f"Available: {list(pool.keys())}"
            )

        model_cfg = pool[model_key]
        llm = self._create_llm(model_cfg)
        self._cache[model_key] = llm

        logger.info(
            "LLMPool: initialized '%s' (%s @ %s)",
            model_key, model_cfg["model"], model_cfg.get("base_url", "default"),
        )
        return llm

    def get_all_keys(self) -> list:
        """List all available model keys in the pool."""
        return list(self.config.get("llm_pool", {}).keys())

    def get_role_mapping(self) -> Dict[str, str]:
        """Return current role → model_key mapping."""
        return dict(self.config.get("llm_roles", {}))

    def _create_llm(self, model_cfg: Dict[str, Any]) -> Any:
        """Create an LLM client from a pool entry config."""
        provider = model_cfg.get("provider", "openai")
        model = model_cfg["model"]
        base_url = model_cfg.get("base_url")

        # Resolve API key from env var or direct value
        api_key = None
        api_key_env = model_cfg.get("api_key_env")
        if api_key_env:
            api_key = os.environ.get(api_key_env)
        if not api_key:
            api_key = model_cfg.get("api_key")

        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if self.callbacks:
            kwargs["callbacks"] = self.callbacks

        # Forward provider-specific settings
        for key in ("reasoning_effort", "thinking_level", "effort", "timeout"):
            if key in model_cfg:
                kwargs[key] = model_cfg[key]

        client = create_llm_client(
            provider=provider,
            model=model,
            base_url=base_url,
            **kwargs,
        )
        return client.get_llm()

    def _get_legacy_llm(self, tier: str) -> Any:
        """Fallback: create LLM from legacy deep_think_llm/quick_think_llm config."""
        if tier in self._legacy_cache:
            return self._legacy_cache[tier]

        if tier == "deep":
            model = self.config.get("deep_think_llm")
        else:
            model = self.config.get("quick_think_llm")

        kwargs = {}
        if self.callbacks:
            kwargs["callbacks"] = self.callbacks

        # Provider-specific thinking config
        provider = self.config.get("llm_provider", "openai")
        for cfg_key, kwarg_key in [
            ("google_thinking_level", "thinking_level"),
            ("openai_reasoning_effort", "reasoning_effort"),
            ("anthropic_effort", "effort"),
        ]:
            val = self.config.get(cfg_key)
            if val:
                kwargs[kwarg_key] = val

        client = create_llm_client(
            provider=provider,
            model=model,
            base_url=self.config.get("backend_url"),
            **kwargs,
        )
        llm = client.get_llm()
        self._legacy_cache[tier] = llm
        return llm
