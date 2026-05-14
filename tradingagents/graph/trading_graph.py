# TradingAgents/graph/trading_graph.py

import os
from pathlib import Path
import json
from datetime import date
from typing import Dict, Any, Tuple, List, Optional

from langgraph.prebuilt import ToolNode

from tradingagents.llm_clients import create_llm_client
from tradingagents.llm_clients.pool import LLMPool
from tradingagents.llm_clients.probe import LLMProbe

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.interface import route_to_vendor

# Import the new abstract tool methods from agent_utils
from tradingagents.agents.utils.agent_utils import (
    get_stock_data,
    get_indicators,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_news,
    get_insider_transactions,
    get_global_news
)

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor


def get_overnight_candidates(trade_date: str, top_k: int = 20):
    """Graph tool wrapper for overnight candidate retrieval."""
    return route_to_vendor("get_overnight_candidates", trade_date, top_k)



def get_overnight_candidate_summary(trade_date: str, top_k: int = 20):
    """Graph tool wrapper for overnight candidate summaries."""
    return route_to_vendor("get_overnight_candidate_summary", trade_date, top_k)



def get_overnight_candidate_payload(trade_date: str, top_k: int = 20):
    """Graph tool wrapper for overnight candidate payload text."""
    return route_to_vendor("get_overnight_candidate_payload", trade_date, top_k)


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
        callbacks: Optional[List] = None,
        run_probe: bool = True,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
            callbacks: Optional list of callback handlers (e.g., for tracking LLM/tool stats)
            run_probe: If True (default), probe all models at startup and
                       schedule roles dynamically based on results.
                       Even when False, cost-aware fallback ensures correct
                       model-to-role matching (just without live probing).
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.callbacks = callbacks or []

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # Initialize LLM Pool (multi-model, role-based, mode-aware)
        self.llm_pool = LLMPool(self.config, callbacks=self.callbacks)
        self.bootstrap_error: Optional[str] = None

        # Probe models and schedule roles dynamically.
        # Always attempt probe+schedule; fall back gracefully on failure.
        self._probe_and_schedule(run_probe, debug)

        self.deep_thinking_llm = None
        self.quick_thinking_llm = None
        self.graph_setup = None
        self.graph = None

        try:
            # Legacy compatibility: expose deep/quick for components that use them
            self.deep_thinking_llm = self.llm_pool.get_llm("research_manager")
            self.quick_thinking_llm = self.llm_pool.get_llm("market_analyst")
        except Exception as exc:
            self.bootstrap_error = f"LLM bootstrap unavailable: {exc}"
            if self.debug:
                print(f"[TradingAgentsGraph] {self.bootstrap_error}")
        
        # Initialize memories
        self.bull_memory = FinancialSituationMemory("bull_memory", self.config)
        self.bear_memory = FinancialSituationMemory("bear_memory", self.config)
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
        self.portfolio_manager_memory = FinancialSituationMemory("portfolio_manager_memory", self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
        )
        if self.deep_thinking_llm is not None and self.quick_thinking_llm is not None:
            self.graph_setup = GraphSetup(
                llm_pool=self.llm_pool,
                quick_thinking_llm=self.quick_thinking_llm,
                deep_thinking_llm=self.deep_thinking_llm,
                tool_nodes=self.tool_nodes,
                bull_memory=self.bull_memory,
                bear_memory=self.bear_memory,
                trader_memory=self.trader_memory,
                invest_judge_memory=self.invest_judge_memory,
                portfolio_manager_memory=self.portfolio_manager_memory,
                conditional_logic=self.conditional_logic,
            )

        self.propagator = Propagator()
        self.reflector = Reflector(
            llm_pool=self.llm_pool,
            quick_thinking_llm=self.quick_thinking_llm,
        )
        self.signal_processor = SignalProcessor(
            llm_pool=self.llm_pool,
            quick_thinking_llm=self.quick_thinking_llm,
        )

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph when LLM bootstrap succeeded
        if self.graph_setup is not None:
            self.graph = self.graph_setup.setup_graph(selected_analysts)

    def _get_provider_kwargs(self) -> Dict[str, Any]:
        """Get provider-specific kwargs for LLM client creation."""
        kwargs = {}
        provider = self.config.get("llm_provider", "").lower()

        if provider == "google":
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level

        elif provider == "openai":
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

        elif provider == "anthropic":
            effort = self.config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort

        return kwargs

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources using abstract methods."""
        return {
            "market": ToolNode(
                [get_stock_data, get_indicators],
                handle_tool_errors=True,
            ),
            "social": ToolNode(
                [get_news],
                handle_tool_errors=True,
            ),
            "news": ToolNode(
                [get_news, get_global_news, get_insider_transactions],
                handle_tool_errors=True,
            ),
            "fundamentals": ToolNode(
                [get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement],
                handle_tool_errors=True,
            ),
            "overnight": ToolNode(
                [
                    get_overnight_candidates,
                    get_overnight_candidate_summary,
                    get_overnight_candidate_payload,
                ],
                handle_tool_errors=True,
            ),
        }

    def propagate(self, company_name, trade_date):
        """Run the trading agents graph for a company on a specific date."""
        if self.graph is None:
            raise RuntimeError(
                "Trading graph is unavailable because LLM bootstrap failed. "
                f"bootstrap_error={self.bootstrap_error}"
            )

        self.ticker = company_name

        # Initialize state
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date
        )
        final_state = self._run_graph_from_state(init_agent_state)

        # Store current state for reflection
        self.curr_state = final_state

        # Log state
        self._log_state(trade_date, final_state)

        # Return decision and processed signal
        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _run_graph_from_state(self, init_agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the compiled graph from an already prepared initial state."""
        if self.graph is None:
            raise RuntimeError(
                "Trading graph is unavailable because LLM bootstrap failed. "
                f"bootstrap_error={self.bootstrap_error}"
            )

        args = self.propagator.get_graph_args()
        if self.debug:
            final_state = dict(init_agent_state)
            for chunk in self.graph.stream(init_agent_state, **args):
                for node_name, delta in chunk.items():
                    if node_name == "__end__":
                        continue
                    for key, val in delta.items():
                        final_state[key] = val
                    new_msgs = delta.get("messages", [])
                    if new_msgs:
                        new_msgs[-1].pretty_print()
            return final_state

        return self.graph.invoke(init_agent_state, **args)

    def propagate_overnight(
        self,
        trade_date: str,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
    ):
        """Run overnight analysis.

        If the full graph is available, overnight state is injected into the
        compiled graph so the normal analyst/research/trader pipeline runs.
        If LLM bootstrap is unavailable, this method falls back to returning
        the prepared overnight bootstrap state.
        """
        candidate_pool_size = int(top_k or self.config.get("overnight_candidate_pool_size", 20))
        target_top_n = int(top_n or self.config.get("overnight_top_n", 5))
        payload = get_overnight_candidate_payload(trade_date, candidate_pool_size)
        summary = get_overnight_candidate_summary(trade_date, candidate_pool_size)
        candidates = get_overnight_candidates(trade_date, candidate_pool_size)

        company_label = f"overnight::{trade_date}"
        self.ticker = company_label
        constraints_json = json.dumps(
            {
                "top_n": target_top_n,
                "candidate_pool_size": candidate_pool_size,
                "filter_variant": self.config.get("overnight_filter_variant", "strict_risk"),
                "weight_variant": self.config.get("overnight_weight_variant", "baseline"),
                "exit_mode": self.config.get("overnight_exit_mode", "open"),
                "allow_cash": self.config.get("overnight_allow_cash", True),
            },
            ensure_ascii=False,
            indent=2,
        )
        init_agent_state = self.propagator.create_initial_overnight_state(
            trade_date=trade_date,
            payload=payload,
            summary_json=json.dumps(summary, ensure_ascii=False, indent=2),
            candidates_json=candidates.to_json(orient="records", force_ascii=False),
            selected_json=candidates.head(target_top_n).to_json(orient="records", force_ascii=False),
            constraints_json=constraints_json,
            company_label=company_label,
        )

        if self.graph is not None:
            final_state = self._run_graph_from_state(init_agent_state)
            self.curr_state = final_state
            self.last_overnight_decision = final_state.get("final_trade_decision", payload)
            try:
                self.last_overnight_processed_signal = self.process_signal(self.last_overnight_decision)
            except Exception:
                self.last_overnight_processed_signal = self.last_overnight_decision
            final_state["last_overnight_decision"] = self.last_overnight_decision
            final_state["last_overnight_processed_signal"] = self.last_overnight_processed_signal
            final_state["overnight_candidate_payload"] = payload
            self._log_state(trade_date, final_state)
            return final_state, payload

        self.curr_state = init_agent_state
        self.last_overnight_decision = payload
        self.last_overnight_processed_signal = payload
        init_agent_state["last_overnight_decision"] = self.last_overnight_decision
        init_agent_state["last_overnight_processed_signal"] = self.last_overnight_processed_signal
        init_agent_state["overnight_candidate_payload"] = payload
        self._log_state(trade_date, init_agent_state)
        return init_agent_state, payload

    def propagate_overnight_live(
        self,
        trade_date: str,
        payload: str,
        candidate_summary: Dict[str, Any] | str,
        candidates_json: str,
        selected_json: str,
        constraints: Dict[str, Any] | str,
        company_label: Optional[str] = None,
    ):
        """Run TradingAgents2.0 graph on a live pre-close candidate buffer.

        This path is for the user's desired workflow: build a Top10/Top15/Top20
        buffer before 14:55, let the agent graph review it, then use the review
        scores during the final 14:55 fusion step.
        """
        label = company_label or f"overnight-live-review::{trade_date}"
        self.ticker = label
        summary_json = candidate_summary if isinstance(candidate_summary, str) else json.dumps(candidate_summary, ensure_ascii=False, indent=2)
        constraints_json = constraints if isinstance(constraints, str) else json.dumps(constraints, ensure_ascii=False, indent=2)
        init_agent_state = self.propagator.create_initial_overnight_state(
            trade_date=trade_date,
            payload=payload,
            summary_json=summary_json,
            candidates_json=candidates_json,
            selected_json=selected_json,
            constraints_json=constraints_json,
            company_label=label,
        )
        init_agent_state["selection_rationale"] = "Live pre-close review buffer: TradingAgents2.0 reviews the Top-K candidates before 14:55 final fusion."
        init_agent_state["final_trade_decision"] = payload

        if self.graph is not None:
            final_state = self._run_graph_from_state(init_agent_state)
            self.curr_state = final_state
            self.last_overnight_decision = final_state.get("final_trade_decision", payload)
            try:
                self.last_overnight_processed_signal = self.process_signal(self.last_overnight_decision)
            except Exception:
                self.last_overnight_processed_signal = self.last_overnight_decision
            final_state["last_overnight_decision"] = self.last_overnight_decision
            final_state["last_overnight_processed_signal"] = self.last_overnight_processed_signal
            final_state["overnight_candidate_payload"] = payload
            self._log_state(trade_date, final_state)
            return final_state, self.last_overnight_decision

        self.curr_state = init_agent_state
        self.last_overnight_decision = payload
        self.last_overnight_processed_signal = payload
        init_agent_state["last_overnight_decision"] = payload
        init_agent_state["last_overnight_processed_signal"] = payload
        init_agent_state["overnight_candidate_payload"] = payload
        self._log_state(trade_date, init_agent_state)
        return init_agent_state, payload

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "strategy_mode": final_state.get("strategy_mode", "single_stock"),
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "global_market_context": final_state.get("global_market_context", ""),
            "overnight_context": final_state.get("overnight_context", ""),
            "candidate_universe_summary": final_state.get("candidate_universe_summary", ""),
            "candidate_snapshot": final_state.get("candidate_snapshot", ""),
            "baseline_reference_picks": final_state.get("baseline_reference_picks", ""),
            "strict_risk_reference_picks": final_state.get("strict_risk_reference_picks", ""),
            "screened_candidates": final_state.get("screened_candidates", ""),
            "selected_candidates": final_state.get("selected_candidates", ""),
            "rejected_candidates": final_state.get("rejected_candidates", ""),
            "selection_constraints": final_state.get("selection_constraints", ""),
            "selection_rationale": final_state.get("selection_rationale", ""),
            "override_reasons": final_state.get("override_reasons", ""),
            "rejected_reason_map": final_state.get("rejected_reason_map", ""),
            "portfolio_construction_plan": final_state.get("portfolio_construction_plan", ""),
            "final_portfolio": final_state.get("final_portfolio", ""),
            "overnight_candidate_payload": final_state.get("overnight_candidate_payload", ""),
            "last_overnight_decision": final_state.get("last_overnight_decision", ""),
            "last_overnight_processed_signal": final_state.get("last_overnight_processed_signal", ""),
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state.get("trader_investment_plan", ""),
            "risk_debate_state": {
                "aggressive_history": final_state["risk_debate_state"]["aggressive_history"],
                "conservative_history": final_state["risk_debate_state"]["conservative_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state.get("investment_plan", ""),
            "final_trade_decision": final_state.get("final_trade_decision", ""),
        }

        # Save to file
        directory = Path(f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/")
        directory.mkdir(parents=True, exist_ok=True)

        with open(
            f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/full_states_log_{trade_date}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(self.log_states_dict, f, indent=4)

    def reflect_and_remember(self, returns_losses):
        """Reflect on decisions and update memory based on returns."""
        self.reflector.reflect_bull_researcher(
            self.curr_state, returns_losses, self.bull_memory
        )
        self.reflector.reflect_bear_researcher(
            self.curr_state, returns_losses, self.bear_memory
        )
        self.reflector.reflect_trader(
            self.curr_state, returns_losses, self.trader_memory
        )
        self.reflector.reflect_invest_judge(
            self.curr_state, returns_losses, self.invest_judge_memory
        )
        self.reflector.reflect_portfolio_manager(
            self.curr_state, returns_losses, self.portfolio_manager_memory
        )

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)

    # ------------------------------------------------------------------
    # Probe & schedule helpers
    # ------------------------------------------------------------------

    def _probe_and_schedule(self, run_probe: bool, debug: bool) -> None:
        """Probe models and schedule roles.

        When *run_probe* is True (default), every model in the pool is
        tested for connectivity and capabilities, then roles are assigned
        based on probe scores + cost preferences.

        When *run_probe* is False the probe is skipped, but we still call
        ``schedule_roles`` with **synthetic probe results** derived from
        the pool config so that cost-aware scheduling still happens.
        This avoids the old bug where skipping the probe caused every
        role to silently fall back to the first model in the dict.
        """
        import logging
        _log = logging.getLogger(__name__)

        if run_probe:
            try:
                probe = LLMProbe(self.config)
                probe_results = probe.probe_all(verbose=debug)
                assignments = self.llm_pool.schedule_roles(probe_results)
                if debug:
                    print("\n[LLM Schedule]\n"
                          + self.llm_pool.get_schedule_summary())
                return
            except Exception as e:
                _log.warning(
                    "Probe failed (%s), falling back to synthetic schedule", e,
                )
                # Fall through to synthetic scheduling below

        # --- Synthetic scheduling (no live probe) ---
        self._synthetic_schedule(debug)

    def _synthetic_schedule(self, debug: bool) -> None:
        """Build synthetic ProbeResults from pool config and schedule roles.

        Every model is assumed available with chat + deepthink support.
        Higher cost_tier models get a quality bonus so that ``prefer_cost:
        any`` roles (trader, research_manager, etc.) are assigned to the
        most capable model rather than winning by dict-order tie-break.
        """
        import logging
        from tradingagents.llm_clients.probe import ProbeResult

        _log = logging.getLogger(__name__)
        _log.info("Running synthetic schedule (no live probe)")

        # Quality bonus: higher-cost models are assumed more capable
        _QUALITY_BONUS = {"high": 6.0, "medium": 3.0, "low": 0.0}

        pool = self.config.get("llm_pool", {})
        synthetic: Dict[str, ProbeResult] = {}

        for model_key, model_cfg in pool.items():
            cost_tier = model_cfg.get("cost_tier", "medium")
            pr = ProbeResult(
                model_key=model_key,
                model_name=model_cfg.get("model", ""),
                base_url=model_cfg.get("base_url", ""),
                context_window=model_cfg.get("context_window", 128000),
                available=True,
                chat_ok=True,
                chat_correct=True,
                deepthink_ok=True,
                deepthink_correct=True,
                deepthink_quality="structured",
                capabilities=["chat", "deepthink"],
                cost_tier=cost_tier,
                # Synthetic latency: assume 2s for all
                latency_chat_ms=2000.0,
            )
            pr.compute_score()
            # Add quality bonus after compute_score so it doesn't
            # get capped by the 100-point ceiling inside compute_score
            pr.score = min(100.0, pr.score + _QUALITY_BONUS.get(cost_tier, 0.0))
            synthetic[model_key] = pr

        assignments = self.llm_pool.schedule_roles(synthetic)
        if debug:
            print("\n[LLM Schedule (synthetic)]\n"
                  + self.llm_pool.get_schedule_summary())
