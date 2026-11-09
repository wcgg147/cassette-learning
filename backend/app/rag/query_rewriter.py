"""
Query Rewriter - 口语化查询改写与意图扩展模块 (Query Expansion & Intent Normalization)

将用户口语化的提问（如“那个扫不出来的码”、“怎么讲项目亮点”、“方便面要袋子”）
自动扩展并改写为结构化的检索关键词与范式意图，提高 RAG 召回率 (Recall@K)。
"""

import re
from typing import List, Dict, Any

SPOKEN_MAP = {
    # 扫码/故障类
    "扫不出来": "二维码 扫码失败 故障 QR code network issue code isn't working",
    "卡住了": "网络故障 系统卡顿 解决 loading error delay",
    "找不到码": "二维码 找码 扫码位置 QR code ordering",

    # 餐饮特调/退换
    "少糖": "半糖 少糖 甜度 half sugar less sugar",
    "换牛奶": "燕麦膜 换奶 忌口 oat milk dairy-free",
    "卖光了": "售罄 没货 sold out replacement recommend",

    # 机场/出行
    "行李超重": "行李 托运 超重 overweight bag carry-on",
    "换座位": "靠窗 靠走廊 选座 window seat aisle seat",
    "海关": "海关 入境 停留时间 目的 purpose of visit sightseeing",

    # 职场/面试
    "介绍项目": "面试 自我介绍 项目亮点 STAR法则 spearheaded",
    "讲缺点": "面试 缺点 离职原因 成长心态 weakness growth mindset",
    "打断讲话": "插话 赞同 异议 会议口语 jump in align on",

    # 地道口语/接话
    "同意": "赞同 地道口语替换 You can say that again Spot on",
    "走了": "离开 口语表达 I've got to run Catch you later",
    "明白了": "听懂 短句接话 Gotcha Fair enough Totally",

    # 日语
    "要袋子": "レジ袋 袋はどうしますか 袋一枚お願いします",
    "加热": "温めますか 温めてください",
    "相槌": "そうなんですね なるほど たしかに へえー",
}

# 错误类型/教学概念 → 检索关键词扩展表
# 用于把 TutorAgent 传入的 error_type / weakness / scene_goal 扩展为知识库关键词，
# 确保 SLA 文献块（KN-061~KN-068）及对应语言点块能被稳定召回。
CONCEPT_EXPANSION = {
    # 冠词
    "article_missing": "冠词 article a an the 遗漏 CLEC 中国学习者 错误统计",
    "article": "冠词 article a an the 遗漏 CLEC",
    "en_articles": "冠词 article a an the 遗漏 CLEC",
    # 主谓一致
    "subject_verb": "主谓一致 subject-verb agreement CLEC 单复数 第三人称",
    "subject_verb_agreement": "主谓一致 subject-verb agreement CLEC 单复数 第三人称",
    "en_subject_verb_agreement": "主谓一致 subject-verb agreement CLEC 单复数 第三人称",
    # 时态
    "tense": "时态 tense past present perfect CLEC 错误统计",
    "tense_past": "时态 tense past CLEC 错误统计",
    "tense_present_perfect": "时态 present perfect CLEC 错误统计",
    "tense_future": "时态 future CLEC 错误统计",
    "en_tense": "时态 tense CLEC 错误统计",
    # 介词
    "preposition": "介词 preposition collocation CLEC 搭配错误",
    "prep_wrong": "介词 preposition collocation CLEC 搭配错误",
    "en_prepositions": "介词 preposition collocation CLEC 搭配错误",
    # 词序
    "word_order": "语序 word order 中文母语者迁移 CLEC",
    "en_word_order": "语序 word order 中文母语者迁移 CLEC",
    # 词汇/搭配
    "collocation": "搭配 collocation CLEC 词汇错误 中国学习者",
    "literal_translation": "直译 中式英语 Chinglish CLEC 词汇错误",
    "chinglish": "直译 中式英语 Chinglish CLEC 词汇错误",
    "vocabulary": "词汇错误 CLEC 搭配 collocation",
    # 复数
    "plural_missing": "复数 plural CLEC 语法错误",
    # 修正性反馈
    "corrective_feedback": "corrective feedback recast explicit correction meta-analysis Li 2010 Lyster Saito",
    "recast": "corrective feedback recast explicit correction meta-analysis",
    "explicit_correction": "corrective feedback explicit correction Ellis Loewen Erlam 2006",
    # 教学方法论
    "output_hypothesis": "输出假说 output hypothesis Swain 1985 comprehensible output pushed output",
    "pushed_output": "输出假说 output hypothesis Swain 1985 pushed output",
    "interaction_hypothesis": "互动假说 interaction hypothesis Long 1996 negotiation of meaning clarification request",
    "negotiation_of_meaning": "互动假说 interaction hypothesis Long 1996 negotiation of meaning",
    "formulaic_sequence": "语块教学 formulaic sequences Wray 2002 chunks 程式化语言",
    "formulaic": "语块教学 formulaic sequences Wray 2002 chunks",
    "affective_filter": "情感过滤 affective filter Krashen 1982 anxiety 焦虑",
    "anxiety": "情感过滤 affective filter Krashen 1982 anxiety",
    "scaffolding": "支架式教学 scaffolding Wood Bruner 填空提示 首字母",
    "contrastive_analysis": "对比分析假说 contrastive analysis Lado 1957 L1 transfer 母语迁移",
    "l1_transfer": "对比分析假说 contrastive analysis Lado 1957 L1 transfer 母语迁移",
    "corpus": "CLEC 中国学习者英语语料库 桂诗春 杨惠中 错误统计 empirical_data",
}

class QueryRewriter:
    """口语化查询改写器。"""

    @staticmethod
    def rewrite(query: str) -> str:
        """
        改写口语提问，融合原始查询与扩展的规范专业词库。
        同时把 error_type / 教学概念扩展为知识库关键词，提升 SLA 文献块召回率。
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return cleaned_query

        expansions = []
        lowered = cleaned_query.lower()

        # 1. 口语化映射
        for spoken, formal in SPOKEN_MAP.items():
            if spoken in lowered:
                expansions.append(formal)

        # 2. 错误类型 / 教学概念扩展（按 token 与整句两级匹配）
        query_tokens = set(re.findall(r"[a-z_]+", lowered))
        for concept, formal in CONCEPT_EXPANSION.items():
            # 整句或子串命中（适合中文关键词）
            if concept in lowered:
                expansions.append(formal)
                continue
            # 英文 error_type token 命中（适合 article_missing / en_articles 等）
            if concept in query_tokens:
                expansions.append(formal)

        if expansions:
            expanded_str = " ".join(expansions)
            return f"{cleaned_query} {expanded_str}"
        return cleaned_query
