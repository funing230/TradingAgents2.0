import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),

    # =========================================================
    # LLM Pool — register all available models here
    # Each entry: provider, model name, base_url, api_key source
    # Add new models without changing any code
    # =========================================================
    "llm_pool": {
        "claude-opus": {
            "provider": "openai",
            "model": "claude-opus-4-6-thinking",
            "base_url": "https://www.fucheers.top/v1",
            "api_key_env": "CLAUDE_API_KEY",
            "context_window": 200000,
            "max_tokens": 16384,
        },
        "gpt54": {
            "provider": "openai",
            "model": "gpt-5.4",
            "base_url": "http://92scw.cn/v1",
            "api_key_env": "GPT54_API_KEY",
            "context_window": 128000,
            "max_tokens": 8192,
        },
        "gemini": {
            "provider": "openai",
            "model": "[L]gemini-3-pro-preview",
            "base_url": "https://new.lemonapi.site/v1",
            "api_key_env": "GEMINI_API_KEY",
            "context_window": 2000000,
            "max_tokens": 32768,
        },
        # === 预留扩展位 ===
        # "deepseek": {
        #     "provider": "openai",
        #     "model": "deepseek-chat",
        #     "base_url": "https://api.deepseek.com/v1",
        #     "api_key_env": "DEEPSEEK_API_KEY",
        #     "context_window": 16000,
        #     "max_tokens": 4096,
        # },
        # "qwen": {
        #     "provider": "openai",
        #     "model": "qwen-max",
        #     "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        #     "api_key_env": "QWEN_API_KEY",
        #     "context_window": 128000,
        #     "max_tokens": 8192,
        # },
        # "deepseek-r1": {
        #     "provider": "openai",
        #     "model": "deepseek-reasoner",
        #     "base_url": "https://api.deepseek.com/v1",
        #     "api_key_env": "DEEPSEEK_API_KEY",
        #     "context_window": 64000,
        #     "max_tokens": 8192,
        # },
    },

    # =========================================================
    # Role → Model mapping
    # Map each agent role to a model key from llm_pool
    # Change assignments without touching any agent code
    # =========================================================
    "llm_roles": {
        "market_analyst":       "gpt54",
        "news_analyst":         "gpt54",
        "social_analyst":       "gpt54",
        "fundamentals_analyst": "gpt54",
        "bull_researcher":      "gpt54",
        "bear_researcher":      "gpt54",
        "research_manager":     "claude-opus",    # 主管决策
        "trader":               "gpt54",
        "aggressive_debater":   "gemini",
        "conservative_debater": "gemini",
        "neutral_debater":      "gemini",
        "portfolio_manager":    "claude-opus",    # 最终决策
    },

    # =========================================================
    # Legacy LLM settings (backward compatible fallback)
    # Used when llm_roles is empty or a role is not mapped
    # =========================================================
    "llm_provider": "openai",
    "deep_think_llm": "claude-opus-4-6-thinking",
    "quick_think_llm": "gpt-5.4",
    "backend_url": "http://49.51.249.22/v1",

    # Provider-specific thinking configuration
    "google_thinking_level": None,
    "openai_reasoning_effort": None,
    "anthropic_effort": None,

    # Output language for analyst reports and final decision
    "output_language": "English",

    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,

    # Data vendor configuration
    "data_vendors": {
        "core_stock_apis": "tushare,yfinance",
        "technical_indicators": "tushare,yfinance",
        "fundamental_data": "tushare,yfinance",
        "news_data": "akshare,yfinance",
    },
    "tool_vendors": {},
}
