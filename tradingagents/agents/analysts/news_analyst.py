from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    build_global_context_block,
    get_global_news,
    get_language_instruction,
    get_news,
)
from tradingagents.dataflows.config import get_config


def _build_overnight_brief(state):
    return (
        "\n\n--- OVERNIGHT CANDIDATE CONTEXT ---\n"
        + str(state.get("overnight_context", ""))
        + "\n\nCandidate summary:\n"
        + str(state.get("candidate_universe_summary", ""))
        + "\n\nSelected candidates:\n"
        + str(state.get("selected_candidates", state.get("final_portfolio", "")))
        + "\n--- END OVERNIGHT CANDIDATE CONTEXT ---\n"
    )


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])
        global_context = build_global_context_block(state)
        is_overnight = str(state.get("strategy_mode", "single_stock") or "single_stock") == "overnight"

        tools = [] if is_overnight else [
            get_news,
            get_global_news,
        ]

        system_message = (
            "You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Use the available tools: get_news(query, start_date, end_date) for company-specific or targeted news searches, and get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + (" In overnight mode, do not call news tools for the synthetic label. Instead, analyze the provided overnight candidate context and global market context directly, and write a news-style report for the overnight basket." if is_overnight else "")
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}{global_context}{overnight_brief}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)
        prompt = prompt.partial(global_context=global_context)
        prompt = prompt.partial(overnight_brief=_build_overnight_brief(state) if is_overnight else "")

        if is_overnight:
            rendered = prompt.invoke({"messages": state["messages"]})
            result = llm.invoke(rendered.to_messages())
        else:
            chain = prompt | llm.bind_tools(tools)
            result = chain.invoke(state["messages"])

        report = result.content

        if not is_overnight and len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
