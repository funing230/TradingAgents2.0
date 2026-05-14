# TradingAgents/graph/propagation.py

from typing import Dict, Any, List, Optional
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)


class Propagator:
    """Handles state initialization and propagation through the graph."""

    def __init__(self, max_recur_limit=100):
        """Initialize with configuration parameters."""
        self.max_recur_limit = max_recur_limit

    def _empty_invest_debate_state(self) -> InvestDebateState:
        return InvestDebateState(
            {
                "bull_history": "",
                "bear_history": "",
                "history": "",
                "current_response": "",
                "judge_decision": "",
                "count": 0,
            }
        )

    def _empty_risk_debate_state(self) -> RiskDebateState:
        return RiskDebateState(
            {
                "aggressive_history": "",
                "conservative_history": "",
                "neutral_history": "",
                "history": "",
                "latest_speaker": "",
                "current_aggressive_response": "",
                "current_conservative_response": "",
                "current_neutral_response": "",
                "judge_decision": "",
                "count": 0,
            }
        )

    def _base_state(self, company_name: str, trade_date: str) -> Dict[str, Any]:
        """Create base state shared by single-stock and overnight modes."""
        return {
            "messages": [("human", company_name)],
            "company_of_interest": company_name,
            "trade_date": str(trade_date),
            "strategy_mode": "single_stock",
            "investment_debate_state": self._empty_invest_debate_state(),
            "risk_debate_state": self._empty_risk_debate_state(),
            "market_report": "",
            "fundamentals_report": "",
            "sentiment_report": "",
            "news_report": "",
            "global_market_context": "",
            "overnight_context": "",
            "candidate_universe_summary": "",
            "candidate_snapshot": "",
            "baseline_reference_picks": "",
            "strict_risk_reference_picks": "",
            "screened_candidates": "",
            "selected_candidates": "",
            "rejected_candidates": "",
            "selection_constraints": "",
            "selection_rationale": "",
            "override_reasons": "",
            "rejected_reason_map": "",
            "portfolio_construction_plan": "",
            "final_portfolio": "",
            "investment_plan": "",
            "trader_investment_plan": "",
            "final_trade_decision": "",
        }

    def create_initial_state(
        self, company_name: str, trade_date: str
    ) -> Dict[str, Any]:
        """Create the initial state for the single-stock agent graph."""
        return self._base_state(company_name, trade_date)

    def create_initial_overnight_state(
        self,
        trade_date: str,
        payload: str,
        summary_json: str,
        candidates_json: str,
        selected_json: str,
        constraints_json: str,
        company_label: str | None = None,
    ) -> Dict[str, Any]:
        """Create the initial state for overnight candidate-driven execution."""
        company_name = company_label or f"overnight::{trade_date}"
        state = self._base_state(company_name, trade_date)
        state.update(
            {
                "strategy_mode": "overnight",
                "messages": [("human", payload)],
                "company_of_interest": company_name,
                "overnight_context": payload,
                "candidate_universe_summary": summary_json,
                "candidate_snapshot": candidates_json,
                "screened_candidates": candidates_json,
                "selected_candidates": selected_json,
                "selection_constraints": constraints_json,
                "portfolio_construction_plan": selected_json,
                "final_portfolio": selected_json,
                "selection_rationale": "Overnight runtime bootstrap: selected top-ranked candidates from the configured provider output.",
                "rejected_candidates": "[]",
                "rejected_reason_map": "{}",
                "final_trade_decision": payload,
            }
        )
        return state

    def get_graph_args(self, callbacks: Optional[List] = None) -> Dict[str, Any]:
        """Get arguments for the graph invocation.

        Args:
            callbacks: Optional list of callback handlers for tool execution tracking.
                       Note: LLM callbacks are handled separately via LLM constructor.
        """
        config = {"recursion_limit": self.max_recur_limit}
        if callbacks:
            config["callbacks"] = callbacks
        return {
            "stream_mode": "updates",
            "config": config,
        }
