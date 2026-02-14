"""
OpenAI Model Access Checker
============================
Tests which models you can actually call with your current API key.
Usage:
    export OPENAI_API_KEY='your-key'
    python check_model_access.py
"""

import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

MODELS_TO_TEST = [
    # GPT-5 family
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    # GPT-4.1 family
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    # GPT-4o family
    "gpt-4o",
    "gpt-4o-mini",
    # Reasoning models
    "o3",
    "o4-mini",
    # Legacy
    "gpt-3.5-turbo",
]

TEST_MESSAGE = [{"role": "user", "content": "Say OK"}]

print(f"\n{'Model':<25} {'Status':<10} {'Detail'}")
print("-" * 65)

accessible = []
blocked = []

for model in MODELS_TO_TEST:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=TEST_MESSAGE,
            max_tokens=5,
        )
        accessible.append(model)
        print(f"✅ {model:<23} {'OK':<10} replied: {response.choices[0].message.content.strip()!r}")

    except Exception as e:
        err = str(e)
        # Extract just the key part of the error
        if "does not exist" in err or "not found" in err:
            reason = "model not found"
        elif "access" in err.lower() or "permission" in err.lower() or "403" in err:
            reason = "no access / tier too low"
        elif "Unsupported" in err or "400" in err:
            reason = "unsupported / bad request"
        elif "429" in err:
            reason = "rate limited"
        else:
            reason = err[:50]

        blocked.append(model)
        print(f"❌ {model:<23} {'BLOCKED':<10} {reason}")

print("\n" + "=" * 65)
print(f"✅ Accessible ({len(accessible)}): {', '.join(accessible)}")
print(f"❌ Blocked    ({len(blocked)}): {', '.join(blocked)}")
print()
