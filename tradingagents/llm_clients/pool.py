"""
LLM Pool — Multi-model manager with role-based assignment and failover.

Manages a pool of LLM instances, each with its own provider/endpoint/key.
Agents request LLMs by role name; the pool resolves to the configured model
and mode (chat or deepthink).

Features:
  - Role → model+mode mapping from config
  - Lazy initialization with cache keyed by model_key:mode
  - Automatic failover via ResilientLLM wrapper
  - Probe integration: disable unavailable models, downgrade missing deepthink
  - Backward compatible: plain string role values treated as chat mode

Usage:
    pool = LLMPool(config)
    llm = pool.get_llm("market_analyst")      # role-based (chat)
    llm = pool.get_llm("research_manager")    # role-based (deepthink + fallback)
    llm = pool.get_llm_by_key("claude-opus", mode="deepthink")  # direct
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Set

from tradingagents.llm_clients import create_llm_client

logger = logging.getLogger(__name__)

# Default role → tier fallback (used when llm_roles is empty)
_ROLE_TIER = {
    "research_manager": "deep",
    "portfolio_manager": "deep",
    "reflector": "deep",
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
    "signal_processor": "quick",
}

_TRANSIENT_KEYWORDS = (
    "timeout", "connection", "reset", "refused", "remote",
    "rate limit", "too many requests", "429", "503", "502",
)


class ResilientLLM:
    """LLM wrapper with automatic retry and failover.

    Tries the primary model first. On transient errors, retries up to
    max_retries times, then moves to the next fallback model.
    Non-transient errors skip retries and move to the next model immediately.
    """

    def __init__(self, primary, fallbacks: List = None, max_retries: int = 2):
        self.primary = primary
        self.fallbacks = fallbacks or []
        self.max_retries = max_retries

    def invoke(self, *args, **kwargs):
        chain = [self.primary] + self.fallbacks
        last_error = None
        for i, llm in enumerate(chain):
            for attempt in range(self.max_retries):
                try:
                    return llm.invoke(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if self._is_transient(e):
                        wait = min(2 ** attempt, 8)
                        logger.warning(
                            "Transient error on model %d attempt %d, retry in %ds: %s",
                            i, attempt + 1, wait, e,
                        )
                        time.sleep(wait)
                        continue
                    # Non-transient: skip to next model
                    logger.warning("Non-transient error on model %d: %s", i, e)
                    break
        raise last_error

    @staticmethod
    def _is_transient(e: Exception) -> bool:
        msg = str(e).lower()
        return any(kw in msg for kw in _TRANSIENT_KEYWORDS)

    # Forward common attributes so downstream code can inspect the primary
    def __getattr__(self, name):
        return getattr(self.primary, name)


class LLMPool:
    """Multi-model pool with role-based LLM assignment, mode support, and failover.

    Config format for llm_roles (new):
        "research_manager": {
            "model": "claude-opus",
            "mode": "deepthink",
            "fallback": [{"model": "gpt54", "mode": "deepthink"}]
        }

    Config format for llm_roles (legacy, still supported):
        "research_manager": "claude-opus"
    """

    def __init__(self, config: Dict[str, Any], callbacks: list = None):
        self.config = config
        self.callbacks = callbacks or []
        self._cache: Dict[str, Any] = {}          # "model_key:mode" → LLM instance
        self._legacy_cache: Dict[str, Any] = {}   # "deep"/"quick" → LLM
        self._disabled_models: Set[str] = set()    # models marked unavailable by probe
        self._no_deepthink: Set[str] = set()       # models without deepthink capability

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_llm(self, role: str) -> Any:
        """Get LLM instance for a specific agent role.

        Resolution order:
          1. config["llm_roles"][role] → model_key + mode → build with fallback
          2. Fallback to legacy deep_think_llm / quick_think_llm
        """
        llm_roles = self.config.get("llm_roles", {})
        role_cfg = llm_roles.get(role)

        if role_cfg is not None:
            # New dict format: {"model": "...", "mode": "...", "fallback": [...]}
            if isinstance(role_cfg, dict):
                return self._build_role_llm(role_cfg)
            # Legacy string format: just a model_key, default to chat
            return self._build_role_llm({"model": role_cfg, "mode": "chat"})

        # No mapping at all → legacy fallback
        tier = _ROLE_TIER.get(role, "quick")
        return self._get_legacy_llm(tier)

    def get_llm_by_key(self, model_key: str, mode: str = "chat") -> Any:
        """Get LLM instance by model pool key and mode."""
        cache_key = f"{model_key}:{mode}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        pool = self.config.get("llm_pool", {})
        if model_key not in pool:
            raise ValueError(
                f"Model '{model_key}' not found in llm_pool. "
                f"Available: {list(pool.keys())}"
            )

        # If model is disabled by probe, raise so caller can handle
        if model_key in self._disabled_models:
            raise RuntimeError(f"Model '{model_key}' is unavailable (probe failed)")

        # If deepthink requested but model can't do it, downgrade to chat
        actual_mode = mode
        if mode == "deepthink" and model_key in self._no_deepthink:
            logger.warning(
                "Model '%s' has no deepthink capability, downgrading to chat", model_key
            )
            actual_mode = "chat"
            cache_key = f"{model_key}:{actual_mode}"
            if cache_key in self._cache:
                return self._cache[cache_key]

        model_cfg = pool[model_key]
        llm = self._create_llm(model_cfg, mode=actual_mode)
        self._cache[cache_key] = llm

        logger.info(
            "LLMPool: initialized '%s' mode=%s (%s @ %s)",
            model_key, actual_mode, model_cfg["model"], model_cfg.get("base_url", "default"),
        )
        return llm

    def get_all_keys(self) -> list:
        """List all available model keys in the pool."""
        return list(self.config.get("llm_pool", {}).keys())

    def get_role_mapping(self) -> Dict[str, Any]:
        """Return current role → config mapping."""
        return dict(self.config.get("llm_roles", {}))

    def apply_probe_results(self, results: Dict) -> None:
        """Apply probe results to disable unavailable models and flag missing deepthink.

        Args:
            results: Dict of model_key → ProbeResult from LLMProbe.probe_all()
        """
        for model_key, result in results.items():
            if not result.available:
                self._disabled_models.add(model_key)
                logger.warning("Probe: model '%s' unavailable, disabled for failover", model_key)
            elif not result.deepthink_ok:
                self._no_deepthink.add(model_key)
                logger.warning("Probe: model '%s' has no deepthink, will downgrade to chat", model_key)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_role_llm(self, role_cfg: Dict) -> Any:
        """Build an LLM (possibly wrapped in ResilientLLM) from a role config."""
        model_key = role_cfg["model"]
        mode = role_cfg.get("mode", "chat")
        fallback_cfgs = role_cfg.get("fallback", [])

        # Try to build primary
        primary = None
        primary_error = None
        if model_key not in self._disabled_models:
            try:
                primary = self.get_llm_by_key(model_key, mode=mode)
            except Exception as e:
                primary_error = e
                logger.warning("Failed to create primary '%s:%s': %s", model_key, mode, e)

        # Build fallbacks
        fallbacks = []
        for fb_cfg in fallback_cfgs:
            fb_key = fb_cfg["model"]
            fb_mode = fb_cfg.get("mode", "chat")
            if fb_key in self._disabled_models:
                continue
            try:
                fb_llm = self.get_llm_by_key(fb_key, mode=fb_mode)
                fallbacks.append(fb_llm)
            except Exception as e:
                logger.warning("Failed to create fallback '%s:%s': %s", fb_key, fb_mode, e)

        if primary is None and not fallbacks:
            raise RuntimeError(
                f"No available model for role config {role_cfg}. "
                f"Primary error: {primary_error}"
            )

        # If no fallbacks, return primary directly (no wrapper overhead)
        if primary and not fallbacks:
            return primary

        # If primary failed but we have fallbacks, use first fallback as primary
        if primary is None:
            primary = fallbacks.pop(0)

        return ResilientLLM(primary, fallbacks)

    def _create_llm(self, model_cfg: Dict[str, Any], mode: str = "chat") -> Any:
        """Create an LLM client from a pool entry config with mode-specific kwargs."""
        provider = model_cfg.get("provider", "openai")
        model = model_cfg["model"]
        base_url = model_cfg.get("base_url")

        # Resolve API key
        api_key = self._resolve_api_key(model_cfg)

        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if self.callbacks:
            kwargs["callbacks"] = self.callbacks

        # Forward base-level provider settings
        for key in ("timeout", "max_tokens"):
            if key in model_cfg:
                kwargs[key] = model_cfg[key]

        # Merge mode-specific kwargs (overrides base)
        modes = model_cfg.get("modes", {})
        mode_kwargs = modes.get(mode, {})
        kwargs.update(mode_kwargs)

        client = create_llm_client(
            provider=provider,
            model=model,
            base_url=base_url,
            **kwargs,
        )
        return client.get_llm()

    @staticmethod
    def _resolve_api_key(model_cfg: Dict) -> Optional[str]:
        """Resolve API key from env var or direct value."""
        import os
        api_key = None
        api_key_env = model_cfg.get("api_key_env")
        if api_key_env:
            api_key = os.environ.get(api_key_env)
        if not api_key:
            api_key = model_cfg.get("api_key")
        return api_key

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
