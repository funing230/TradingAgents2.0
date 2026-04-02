# TradingAgents/graph/conditional_logic.py

import logging
from langchain_core.messages import ToolMessage
from tradingagents.agents.utils.agent_states import AgentState

logger = logging.getLogger(__name__)

# Maximum number of tool-call rounds per analyst before forcing report generation.
_MAX_TOOL_ROUNDS = 10


class ConditionalLogic:
    """Handles conditional logic for determining graph flow."""

    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1):
        """Initialize with configuration parameters."""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    @staticmethod
    def _tool_rounds(messages) -> int:
        """Count how many tool-call round-trips have occurred in current messages."""
        return sum(1 for m in messages if isinstance(m, ToolMessage))

    def should_continue_market(self, state: AgentState):
        """Determine if market analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            if self._tool_rounds(messages) >= _MAX_TOOL_ROUNDS:
                logger.warning("Market analyst hit tool-call limit (%d), forcing report.", _MAX_TOOL_ROUNDS)
                return "Msg Clear Market"
            return "tools_market"
        return "Msg Clear Market"

    def should_continue_social(self, state: AgentState):
        """Determine if social media analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            if self._tool_rounds(messages) >= _MAX_TOOL_ROUNDS:
                logger.warning("Social analyst hit tool-call limit (%d), forcing report.", _MAX_TOOL_ROUNDS)
                return "Msg Clear Social"
            return "tools_social"
        return "Msg Clear Social"

    def should_continue_news(self, state: AgentState):
        """Determine if news analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            if self._tool_rounds(messages) >= _MAX_TOOL_ROUNDS:
                logger.warning("News analyst hit tool-call limit (%d), forcing report.", _MAX_TOOL_ROUNDS)
                return "Msg Clear News"
            return "tools_news"
        return "Msg Clear News"

    def should_continue_fundamentals(self, state: AgentState):
        """Determine if fundamentals analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            if self._tool_rounds(messages) >= _MAX_TOOL_ROUNDS:
                logger.warning("Fundamentals analyst hit tool-call limit (%d), forcing report.", _MAX_TOOL_ROUNDS)
                return "Msg Clear Fundamentals"
            return "tools_fundamentals"
        return "Msg Clear Fundamentals"

    def should_continue_debate(self, state: AgentState) -> str:
        """Determine if debate should continue."""

        if (
            state["investment_debate_state"]["count"] >= 2 * self.max_debate_rounds
        ):  # 3 rounds of back-and-forth between 2 agents
            return "Research Manager"
        if state["investment_debate_state"]["current_response"].startswith("Bull"):
            return "Bear Researcher"
        return "Bull Researcher"

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """Determine if risk analysis should continue."""
        if (
            state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds
        ):  # 3 rounds of back-and-forth between 3 agents
            return "Portfolio Manager"
        if state["risk_debate_state"]["latest_speaker"].startswith("Aggressive"):
            return "Conservative Analyst"
        if state["risk_debate_state"]["latest_speaker"].startswith("Conservative"):
            return "Neutral Analyst"
        return "Aggressive Analyst"
