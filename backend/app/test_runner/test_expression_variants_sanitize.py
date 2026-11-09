# -*- coding: utf-8 -*-
"""Temporary test: expression_variants passthrough in ConversationDirector._sanitize."""
import sys

sys.path.insert(0, ".")
from agent.conversation_director import ConversationDirector  # noqa: E402

# 1) keep variants
p1 = ConversationDirector._sanitize({
    "reply_draft": "Sounds great! I can meet you at the lobby around seven.",
    "expression_variants": [
        "See you in the lobby at seven.",
        "Seven works - lobby?",
        "Let's say seven, lobby.",
    ],
})
assert p1["expression_variants"] == [
    "See you in the lobby at seven.",
    "Seven works - lobby?",
    "Let's say seven, lobby.",
], p1["expression_variants"]
print("case1 keep variants OK:", p1["expression_variants"])

# 2) non-list tolerance
p2 = ConversationDirector._sanitize({"expression_variants": "single string variant"})
print("case2 non-list:", p2["expression_variants"])
assert isinstance(p2["expression_variants"], list)

# 3) missing -> empty list
p3 = ConversationDirector._sanitize({"reply_draft": "hi"})
assert p3["expression_variants"] == []
print("case3 default empty OK")

# 4) over-length truncation
p4 = ConversationDirector._sanitize({"expression_variants": ["x" * 300, "ok variant"]})
assert p4["expression_variants"][0].startswith("x") and len(p4["expression_variants"][0]) <= 140
print("case4 truncation OK")

# 5) fallback_plan passes through _sanitize -> empty list present
fb = ConversationDirector.fallback_plan(
    user_text="hello", learner_intent="general_chat",
    language_mode="english_only", scene={},
)
assert fb["expression_variants"] == []
print("case5 fallback OK")

print("ALL SANITIZE TESTS PASS")
