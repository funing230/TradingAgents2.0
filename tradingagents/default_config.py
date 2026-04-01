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
    # "modes" defines per-mode kwargs merged on top of base config
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
            "modes": {
                "chat": {},
                "deepthink": {"reasoning_effort": "high"},
            },
        },
        "gpt54": {
            "provider": "openai",
            "model": "gpt-5.4",
            "base_url": "http://92scw.cn/v1",
            "api_key_env": "GPT54_API_KEY",
            "context_window": 128000,
            "max_tokens": 8192,
            "modes": {
                "chat": {},
                "deepthink": {"reasoning_effort": "high"},
            },
        },
        "gemini": {
            "provider": "openai",
            "model": "[L]gemini-3-pro-preview",
            "base_url": "https://new.lemonapi.site/v1",
            "api_key_env": "GEMINI_API_KEY",
            "context_window": 2000000,
            "max_tokens": 32768,
            "modes": {
                "chat": {},
                "deepthink": {"thinking_level": "high"},
            },
        },
        # === 预留扩展位 ===
        # "deepseek": {
        #     "provider": "openai",
        #     "model": "deepseek-chat",
        #     "base_url": "https://api.deepseek.com/v1",
        #     "api_key_env": "DEEPSEEK_API_KEY",
        #     "context_window": 16000,
        #     "max_tokens": 4096,
        #     "modes": {"chat": {}, "deepthink": {"reasoning_effort": "high"}},
        # },
        # "qwen": {
        #     "provider": "openai",
        #     "model": "qwen-max",
        #     "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        #     "api_key_env": "QWEN_API_KEY",
        #     "context_window": 128000,
        #     "max_tokens": 8192,
        #     "modes": {"chat": {}, "deepthink": {"reasoning_effort": "high"}},
        # },
    },

    # =========================================================
    # Role → Model mapping
    # Each role maps to {"model": <pool_key>, "mode": "chat"|"deepthink"}
    # Optional "fallback" list for automatic failover
    # Legacy format (plain string) is still supported as chat mode
    # =========================================================
    "llm_roles": {
        "market_analyst":       {"model": "gpt54",       "mode": "chat"},
        "news_analyst":         {"model": "gpt54",       "mode": "chat"},
        "social_analyst":       {"model": "gpt54",       "mode": "chat"},
        "fundamentals_analyst": {"model": "gpt54",       "mode": "chat"},
        "bull_researcher":      {"model": "gpt54",       "mode": "chat"},
        "bear_researcher":      {"model": "gpt54",       "mode": "chat"},
        "research_manager":     {"model": "claude-opus",  "mode": "deepthink",
                                 "fallback": [{"model": "gpt54", "mode": "deepthink"}]},
        "trader":               {"model": "gpt54",       "mode": "deepthink"},
        "aggressive_debater":   {"model": "gemini",      "mode": "chat"},
        "conservative_debater": {"model": "gemini",      "mode": "chat"},
        "neutral_debater":      {"model": "gemini",      "mode": "chat"},
        "portfolio_manager":    {"model": "claude-opus",  "mode": "deepthink",
                                 "fallback": [{"model": "gpt54", "mode": "deepthink"}]},
        "reflector":            {"model": "claude-opus",  "mode": "deepthink",
                                 "fallback": [{"model": "gpt54", "mode": "deepthink"}]},
        "signal_processor":     {"model": "gpt54",       "mode": "chat"},
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
