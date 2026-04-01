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
            "cost_tier": "high",
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
            "cost_tier": "medium",
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
            "cost_tier": "low",
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
        #     "cost_tier": "low",
        #     "modes": {"chat": {}, "deepthink": {"reasoning_effort": "high"}},
        # },
        # "qwen": {
        #     "provider": "openai",
        #     "model": "qwen-max",
        #     "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        #     "api_key_env": "QWEN_API_KEY",
        #     "context_window": 128000,
        #     "max_tokens": 8192,
        #     "cost_tier": "low",
        #     "modes": {"chat": {}, "deepthink": {"reasoning_effort": "high"}},
        # },
    },

    # =========================================================
    # Role requirements (declarative)
    #
    # Each role declares WHAT it needs, not WHO provides it:
    #   mode:        "chat" or "deepthink"
    #   prefer_cost: "low" | "medium" | "high" | "any"
    #
    # The scheduler assigns concrete models at startup based on
    # probe results + these requirements. No hardcoded model names.
    #
    # Backward compatible: old format {"model": "xxx", "mode": "..."}
    # and plain strings ("xxx") are still accepted and bypass scheduling.
    # =========================================================
    "llm_roles": {
        # --- Data collection (high frequency, cost sensitive) ---
        "market_analyst":       {"mode": "chat",      "prefer_cost": "low"},
        "news_analyst":         {"mode": "chat",      "prefer_cost": "low"},
        "social_analyst":       {"mode": "chat",      "prefer_cost": "low"},
        "fundamentals_analyst": {"mode": "chat",      "prefer_cost": "low"},

        # --- Research (moderate reasoning, moderate cost) ---
        "bull_researcher":      {"mode": "chat",      "prefer_cost": "medium"},
        "bear_researcher":      {"mode": "chat",      "prefer_cost": "medium"},

        # --- Decision making (deep reasoning, quality first) ---
        "research_manager":     {"mode": "deepthink", "prefer_cost": "any"},
        "trader":               {"mode": "deepthink", "prefer_cost": "any"},
        "portfolio_manager":    {"mode": "deepthink", "prefer_cost": "any"},

        # --- Risk debate (fast exchange, cost sensitive) ---
        "aggressive_debater":   {"mode": "chat",      "prefer_cost": "low"},
        "conservative_debater": {"mode": "chat",      "prefer_cost": "low"},
        "neutral_debater":      {"mode": "chat",      "prefer_cost": "low"},

        # --- Post-processing ---
        "reflector":            {"mode": "deepthink", "prefer_cost": "any"},
        "signal_processor":     {"mode": "chat",      "prefer_cost": "low"},
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
