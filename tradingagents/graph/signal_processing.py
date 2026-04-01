# TradingAgents/graph/signal_processing.py

from typing import Any, Optional


class SignalProcessor:
    """Processes trading signals to extract actionable decisions.

    Supports two initialization modes:
      - New: pass llm_pool for role-based model selection
      - Legacy: pass a single quick_thinking_llm instance
    """

    def __init__(self, llm_pool=None, quick_thinking_llm=None):
        """Initialize with LLM pool or legacy LLM.

        Args:
            llm_pool: LLMPool instance for role-based model selection.
            quick_thinking_llm: Legacy fallback LLM (used when pool is None).
        """
        self.llm_pool = llm_pool
        self._fallback_llm = quick_thinking_llm

    def _get_llm(self) -> Any:
        """Get LLM for signal processing (uses signal_processor role)."""
        if self.llm_pool:
            return self.llm_pool.get_llm("signal_processor")
        return self._fallback_llm

    def process_signal(self, full_signal: str) -> str:
        """
        Process a full trading signal to extract the core decision.

        Args:
            full_signal: Complete trading signal text

        Returns:
            Extracted rating (BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, or SELL)
        """
        llm = self._get_llm()

        messages = [
            (
                "system",
                "You are an efficient assistant that extracts the trading decision from analyst reports. "
                "Extract the rating as exactly one of: BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, SELL. "
                "Output only the single rating word, nothing else.",
            ),
            ("human", full_signal),
        ]

        return llm.invoke(messages).content
