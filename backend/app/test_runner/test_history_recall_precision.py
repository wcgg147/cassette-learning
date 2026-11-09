"""Recall wording should not outrank the subject the user is asking about."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory.history_retrieval import lexical_tokens, rank_history_sparse


def test_recall_scaffolding_does_not_hide_slow_horses():
    candidates = [
        {
            "candidate_id": "generic-name-memory",
            "session_id": "generic-name-memory",
            "searchable": "Alice and Jason talked about names and their friends.",
            "summary": "Names and friends.",
        },
        {
            "candidate_id": "slow-horses-history",
            "session_id": "slow-horses-history",
            "searchable": (
                "User: Slow Horses Season 5 felt slow and the arrangement made some "
                "characters annoying. Season 6 first two episodes felt better."
            ),
            "summary": "Slow Horses: season 5 and season 6 impressions.",
        },
    ]
    ranked = rank_history_sparse(
        candidates,
        "Do you remember what we talked about regarding Slow Horses?",
        top_k=2,
    )
    assert ranked[0]["candidate_id"] == "slow-horses-history", ranked


def test_chinese_recall_scaffolding_leaves_the_named_subject():
    tokens = lexical_tokens("你还记得我们之前聊过流人吗")
    assert any("流人" in token for token in tokens), tokens
    assert not any("记得" in token or "聊过" in token for token in tokens), tokens


if __name__ == "__main__":
    test_recall_scaffolding_does_not_hide_slow_horses()
    test_chinese_recall_scaffolding_leaves_the_named_subject()
    print("PASS: recall scaffolding no longer outranks a named topic")
