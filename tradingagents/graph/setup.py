# TradingAgents/graph/setup.py

from typing import Dict, Any, Optional
import json
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.dataflows.interface import route_to_vendor

from .conditional_logic import ConditionalLogic
from .global_context import create_global_context_collector


def create_overnight_candidate_builder(default_top_k: int = 20, default_top_n: int = 5):
    """Create a lightweight node that prepares overnight candidate context.

    The node is a no-op for legacy single-stock mode.
    For overnight mode it fetches candidate payload/summary from the local
    provider layer and writes graph-ready state fields before analysts run.
    """

    def overnight_candidate_builder(state):
        if state.get("strategy_mode", "single_stock") != "overnight":
            return {
                "overnight_context": state.get("overnight_context", ""),
                "candidate_universe_summary": state.get("candidate_universe_summary", ""),
                "candidate_snapshot": state.get("candidate_snapshot", ""),
                "selection_constraints": state.get("selection_constraints", ""),
            }

        trade_date = str(state.get("trade_date", ""))
        if not trade_date:
            return {
                "overnight_context": "",
                "candidate_universe_summary": "",
                "candidate_snapshot": "",
                "selection_constraints": json.dumps(
                    {"top_n": default_top_n, "candidate_pool_size": default_top_k},
                    ensure_ascii=False,
                ),
            }

        constraints_raw = state.get("selection_constraints", "")
        if constraints_raw:
            try:
                constraints = json.loads(constraints_raw)
                if constraints.get("candidate_source") == "live_preclose_buffer":
                    return {
                        "messages": state.get("messages", []),
                        "overnight_context": state.get("overnight_context", ""),
                        "candidate_universe_summary": state.get("candidate_universe_summary", ""),
                        "candidate_snapshot": state.get("candidate_snapshot", ""),
                        "screened_candidates": state.get("screened_candidates", state.get("candidate_snapshot", "")),
                        "selected_candidates": state.get("selected_candidates", state.get("final_portfolio", "")),
                        "final_portfolio": state.get("final_portfolio", state.get("selected_candidates", "")),
                        "selection_constraints": constraints_raw,
                    }
            except Exception:
                pass

        top_k = default_top_k
        top_n = default_top_n
        if constraints_raw:
            try:
                constraints = json.loads(constraints_raw)
                top_k = int(constraints.get("candidate_pool_size", top_k))
                top_n = int(constraints.get("top_n", top_n))
            except Exception:
                pass

        payload = route_to_vendor("get_overnight_candidate_payload", trade_date, top_k)
        summary = route_to_vendor("get_overnight_candidate_summary", trade_date, top_k)
        candidates = route_to_vendor("get_overnight_candidates", trade_date, top_k)

        selection_constraints = {
            "top_n": top_n,
            "candidate_pool_size": top_k,
        }
        return {
            "messages": [("human", payload)],
            "overnight_context": payload,
            "candidate_universe_summary": json.dumps(summary, ensure_ascii=False, indent=2),
            "candidate_snapshot": candidates.to_json(orient="records", force_ascii=False),
            "screened_candidates": candidates.to_json(orient="records", force_ascii=False),
            "selected_candidates": candidates.head(top_n).to_json(orient="records", force_ascii=False),
            "final_portfolio": candidates.head(top_n).to_json(orient="records", force_ascii=False),
            "selection_constraints": json.dumps(selection_constraints, ensure_ascii=False, indent=2),
        }

    return overnight_candidate_builder


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: ChatOpenAI,
        deep_thinking_llm: ChatOpenAI,
        tool_nodes: Dict[str, ToolNode],
        bull_memory,
        bear_memory,
        trader_memory,
        invest_judge_memory,
        portfolio_manager_memory,
        conditional_logic: ConditionalLogic,
        llm_pool=None,
    ):
        """Initialize with required components.

        Args:
            quick_thinking_llm: Legacy quick LLM (fallback)
            deep_thinking_llm: Legacy deep LLM (fallback)
            tool_nodes: Tool nodes for each analyst type
            bull_memory, bear_memory, ...: Agent memories
            conditional_logic: Conditional logic for graph edges
            llm_pool: Optional LLMPool for role-based model assignment
        """
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.bull_memory = bull_memory
        self.bear_memory = bear_memory
        self.trader_memory = trader_memory
        self.invest_judge_memory = invest_judge_memory
        self.portfolio_manager_memory = portfolio_manager_memory
        self.conditional_logic = conditional_logic
        self.llm_pool = llm_pool

    def _get_llm(self, role: str, default_tier: str = "quick"):
        """Get LLM for a role. Uses pool if available, else legacy fallback."""
        if self.llm_pool:
            return self.llm_pool.get_llm(role)
        return self.deep_thinking_llm if default_tier == "deep" else self.quick_thinking_llm

    def setup_graph(
        self, selected_analysts=["market", "social", "news", "fundamentals"]
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

        # Create analyst nodes (each can use a different model)
        analyst_nodes = {}
        delete_nodes = {}
        tool_nodes = {}

        if "market" in selected_analysts:
            analyst_nodes["market"] = create_market_analyst(
                self._get_llm("market_analyst")
            )
            delete_nodes["market"] = create_msg_delete()
            tool_nodes["market"] = self.tool_nodes["market"]

        if "social" in selected_analysts:
            analyst_nodes["social"] = create_social_media_analyst(
                self._get_llm("social_analyst")
            )
            delete_nodes["social"] = create_msg_delete()
            tool_nodes["social"] = self.tool_nodes["social"]

        if "news" in selected_analysts:
            analyst_nodes["news"] = create_news_analyst(
                self._get_llm("news_analyst")
            )
            delete_nodes["news"] = create_msg_delete()
            tool_nodes["news"] = self.tool_nodes["news"]

        if "fundamentals" in selected_analysts:
            analyst_nodes["fundamentals"] = create_fundamentals_analyst(
                self._get_llm("fundamentals_analyst")
            )
            delete_nodes["fundamentals"] = create_msg_delete()
            tool_nodes["fundamentals"] = self.tool_nodes["fundamentals"]

        # Create researcher and manager nodes (role-specific models)
        bull_researcher_node = create_bull_researcher(
            self._get_llm("bull_researcher"), self.bull_memory
        )
        bear_researcher_node = create_bear_researcher(
            self._get_llm("bear_researcher"), self.bear_memory
        )
        research_manager_node = create_research_manager(
            self._get_llm("research_manager", "deep"), self.invest_judge_memory
        )
        trader_node = create_trader(
            self._get_llm("trader"), self.trader_memory
        )

        # Create risk analysis nodes
        aggressive_analyst = create_aggressive_debator(
            self._get_llm("aggressive_debater")
        )
        neutral_analyst = create_neutral_debator(
            self._get_llm("neutral_debater")
        )
        conservative_analyst = create_conservative_debator(
            self._get_llm("conservative_debater")
        )
        portfolio_manager_node = create_portfolio_manager(
            self._get_llm("portfolio_manager", "deep"), self.portfolio_manager_memory
        )

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add Global Context Collector node (runs before analysts)
        workflow.add_node("Global Context", create_global_context_collector())
        workflow.add_node("Overnight Candidate Builder", create_overnight_candidate_builder())

        # Add analyst nodes to the graph
        for analyst_type, node in analyst_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)
            workflow.add_node(
                f"Msg Clear {analyst_type.capitalize()}", delete_nodes[analyst_type]
            )
            workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])

        # Add other nodes
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        # Define edges
        # Start with Global Context, then Overnight Candidate Builder, then first analyst
        first_analyst = selected_analysts[0]
        workflow.add_edge(START, "Global Context")
        workflow.add_edge("Global Context", "Overnight Candidate Builder")
        workflow.add_edge("Overnight Candidate Builder", f"{first_analyst.capitalize()} Analyst")

        # Connect analysts in sequence
        for i, analyst_type in enumerate(selected_analysts):
            current_analyst = f"{analyst_type.capitalize()} Analyst"
            current_tools = f"tools_{analyst_type}"
            current_clear = f"Msg Clear {analyst_type.capitalize()}"

            # Add conditional edges for current analyst
            workflow.add_conditional_edges(
                current_analyst,
                getattr(self.conditional_logic, f"should_continue_{analyst_type}"),
                [current_tools, current_clear],
            )
            workflow.add_edge(current_tools, current_analyst)

            # Connect to next analyst or to Bull Researcher if this is the last analyst
            if i < len(selected_analysts) - 1:
                next_analyst = f"{selected_analysts[i+1].capitalize()} Analyst"
                workflow.add_edge(current_clear, next_analyst)
            else:
                workflow.add_edge(current_clear, "Bull Researcher")

        # Add remaining edges
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")
        workflow.add_conditional_edges(
            "Aggressive Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Conservative Analyst": "Conservative Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Aggressive Analyst": "Aggressive Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )

        workflow.add_edge("Portfolio Manager", END)

        # Compile and return
        return workflow.compile()
