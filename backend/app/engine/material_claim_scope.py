"""Shared evidence-scope contract for material analysis and conversation."""

MATERIAL_CLAIM_SCOPE_RULE = (
    "结论强度须与证据对应：单个案例只说明该案例及它展示的可能性，不能证明普遍规律或排除其他解释。"
    "比较时先保留各来源自己的因果链：处境约束、他人或制度提供的支持、当事人的行动和结果，只说原文实际展示的部分；再概括共同点与侧重差异。不要把多因素素材压成“内在力量”与“外部支持”等单一对立，也不要推断某人继续行动是为了谁，除非原文明确说明；若提出自己的解释或倾向，须标明是解读并给出文本依据。"
    "一份素材突出某种因素，不表示另一份只靠相反因素；未排除的因素仍可能起作用，不把‘更强调某方面’改成‘完全不是另一方面’。"
    "作品没有提到或没有展示某种解释，不等于作品否定它，也不等于它永远不存在。"
    "某份来源只有标题或来源不明的旧文本时，不能确认整份内容及它与其他来源的关系，不把资料不足当成内容无关。"
    "跨来源比较前，逐份核对共同主题所对应的具体原句、实际发生的变化，以及是否提供了变化原因；缺少其中一项时不把它补成共同结论。比较话题可以成立，但相似话题本身不能证明相同结果或因果。"
    "解读以具体内容为依据，自然保留适用范围；原文明确的事实仍直接说，不为所有句子机械加‘可能’。"
)


def incomplete_primary_source(sources, blocks, query):
    """Return a coverage gap only for the uniquely best-matched source."""
    from engine.material_analyzer import score_blocks
    ranked = []
    for source in sources:
        mid = source.get("material_id")
        texts = [str(block.get("text") or "") for block in blocks
                 if str(block.get("material_id")) == str(mid) and block.get("span_kind") != "comment"]
        score = max(score_blocks(texts, query), default=0)
        ranked.append((score, source))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked or ranked[0][0] <= 0 or (len(ranked) > 1 and ranked[0][0] == ranked[1][0]):
        return {}
    source = ranked[0][1]
    coverage = source.get("source_coverage") or {}
    if coverage.get("kind") != "video" or coverage.get("body_available") is True:
        return {}
    return {"material_id": source.get("material_id"), "title": source.get("title"),
            "body_available": coverage.get("body_available")}


def asserts_absence_in_unverified_video(reply):
    """Detect a content-absence claim, rather than a lack-of-access statement."""
    import re
    for sentence in re.split(r"(?<=[.!?。！？])\s*", str(reply or "")):
        denial = re.search(r"\b(?:the|this)\s+(?:video|clip|film|material|source)\s+(?:doesn['’]t|does not|didn['’]t|did not|never)\b|(?:视频|片子|素材)(?:中|里)?(?:没有|没|从未)", sentence, re.I)
        denial = denial or (re.search(r"\b(?:video|clip|film|source)\b", sentence, re.I) and re.search(r"\b(?:it doesn['’]t|it does not|only says|only shows)\b", sentence, re.I))
        if denial and not re.search(r"available (?:text|excerpt|material)|page (?:text|description)|currently|cannot confirm|can['’]t confirm|未核验|现有|目前|网页|无法确认", sentence, re.I):
            return True
    return False
