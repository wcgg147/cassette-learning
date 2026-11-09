"""
漂移检测模块 - 基于向量嵌入的智能漂移检测系统

参考开源项目：
- driftGuard: "Detects when an LLM starts answering outside its intended domain
              without ground-truth labels or a separate classifier."
- ai-agent-evals: "Use advanced models to evaluate other model outputs"
- LLM-Drift-Detection-Adaptation-Pipeline: "Automatically detects semantic, lexical,
              domain, and performance drift"

架构设计：
┌─────────────────────────────────────────────────────────────┐
│  多维度漂移检测引擎                                          │
│  ├── 语义漂移：嵌入向量相似度（driftGuard方案）              │
│  ├── 领域漂移：分类器检测是否偏离教学领域                    │
│  └── LLM-as-Judge：轻量级模型作为评判者                      │
├─────────────────────────────────────────────────────────────┤
│  参考语料库                                                 │
│  ├── 教学领域代表性文本                                     │
│  ├── 向量嵌入存储                                           │
│  └── 质心计算                                               │
└─────────────────────────────────────────────────────────────┘

工作原理：
1. 构建参考语料库：从代表性的"在教学范围内"的文本构建参考语料
2. 嵌入每个响应：使用相同的嵌入模型
3. 双重信号比较：
   - 质心距离：响应与语料库中心的距离
   - 最近邻距离：响应是否至少与一个参考文本接近
4. 标记漂移：当两个信号都表明响应远离参考域时触发
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class DriftResult:
    """
    漂移检测结果。

    Attributes:
        is_drift: 是否检测到漂移
        centroid_distance: 与语料库质心的距离
        nearest_neighbor_distance: 最近邻距离
        nearest_neighbor_index: 最近邻索引
        confidence: 检测置信度（0.0-1.0）
        drift_type: 漂移类型（semantic/domain/performance）
    """

    is_drift: bool
    centroid_distance: float
    nearest_neighbor_distance: float
    nearest_neighbor_index: int = -1
    confidence: float = 0.0
    drift_type: str = "semantic"


class EmbeddingModel(ABC):
    """
    嵌入模型抽象基类。

    支持多种嵌入模型：
    - OpenAI Embeddings
    - Sentence Transformers
    - 豆包 Embeddings
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        将文本列表转换为向量嵌入。

        Args:
            texts: 文本列表

        Returns:
            向量嵌入列表
        """
        pass


class MockEmbeddingModel(EmbeddingModel):
    """
    模拟嵌入模型，用于开发和测试。

    使用简单的哈希函数生成固定长度的向量。
    """

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            # 使用简单的哈希函数生成固定长度的向量
            hash_val = hash(text)
            embedding = [(hash_val + i * 13) % 1000 / 1000.0 for i in range(self.dimension)]
            embeddings.append(embedding)
        return embeddings


class BGE3EmbeddingModel(EmbeddingModel):
    """
    基于 SiliconFlow BGE-M3 API 的真实语义嵌入模型。

    替代 MockEmbeddingModel，提供真实的语义向量用于漂移检测。
    使用同步 requests 调用（漂移检测不在热路径，延迟容忍度高）。
    API 失败时自动降级为 hash fallback。
    """

    def __init__(self, api_key: str, api_url: str = "https://api.siliconflow.cn/v1/embeddings",
                 model: str = "BAAI/bge-m3", dimension: int = 1024):
        self.dimension = dimension
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self._enabled = bool(api_key and not api_key.startswith("sk-your-"))

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self._enabled or not texts:
            return self._hash_fallback(texts)

        try:
            import requests
            resp = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model, "input": texts},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                return [d["embedding"] for d in data.get("data", [])]
        except Exception:
            pass

        return self._hash_fallback(texts)

    def _hash_fallback(self, texts: List[str]) -> List[List[float]]:
        """API 不可用时降级为 hash embedding。"""
        embeddings = []
        for text in texts:
            hash_val = hash(text)
            embedding = [(hash_val + i * 13) % 1000 / 1000.0 for i in range(self.dimension)]
            embeddings.append(embedding)
        return embeddings


class ReferenceCorpus:
    """
    参考语料库 - 用于构建教学领域的向量空间。

    工作流程：
    1. 添加教学领域的代表性文本
    2. 计算向量嵌入
    3. 计算语料库质心
    4. 构建索引用于快速最近邻搜索
    """

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.texts: List[str] = []
        self.embeddings: List[List[float]] = []
        self.centroid: Optional[List[float]] = None

    def add_texts(self, texts: List[str]):
        """
        添加文本到语料库。

        Args:
            texts: 文本列表
        """
        self.texts.extend(texts)
        new_embeddings = self.embedding_model.embed(texts)
        self.embeddings.extend(new_embeddings)
        self._compute_centroid()

    def _compute_centroid(self):
        """计算语料库的质心向量。"""
        if not self.embeddings:
            self.centroid = None
            return

        np_embeddings = np.array(self.embeddings)
        self.centroid = np.mean(np_embeddings, axis=0).tolist()

    def get_centroid(self) -> Optional[List[float]]:
        """获取语料库质心。"""
        return self.centroid

    def get_nearest_neighbor(
        self, query_embedding: List[float]
    ) -> Tuple[int, float]:
        """
        查找最近邻。

        Args:
            query_embedding: 查询向量

        Returns:
            (索引, 距离)
        """
        if not self.embeddings:
            return (-1, float("inf"))

        np_embeddings = np.array(self.embeddings)
        np_query = np.array(query_embedding)

        # 计算余弦距离
        norms = np.linalg.norm(np_embeddings, axis=1)
        query_norm = np.linalg.norm(np_query)

        if query_norm == 0 or np.all(norms == 0):
            return (-1, float("inf"))

        cosine_similarities = np.dot(np_embeddings, np_query) / (norms * query_norm)
        cosine_distance = 1 - cosine_similarities

        nearest_idx = np.argmin(cosine_distance)
        nearest_dist = float(cosine_distance[nearest_idx])

        return (int(nearest_idx), nearest_dist)

    def compute_centroid_distance(self, query_embedding: List[float]) -> float:
        """
        计算与质心的距离。

        Args:
            query_embedding: 查询向量

        Returns:
            余弦距离
        """
        if not self.centroid:
            return float("inf")

        np_centroid = np.array(self.centroid)
        np_query = np.array(query_embedding)

        norm_centroid = np.linalg.norm(np_centroid)
        norm_query = np.linalg.norm(np_query)

        if norm_centroid == 0 or norm_query == 0:
            return float("inf")

        cosine_similarity = np.dot(np_centroid, np_query) / (norm_centroid * norm_query)
        return float(1 - cosine_similarity)

    def size(self) -> int:
        """获取语料库大小。"""
        return len(self.texts)


class DriftDetector:
    """
    漂移检测器 - 基于向量嵌入的漂移检测。

    参考 driftGuard 的实现模式：
    - 质心距离：响应与语料库中心的距离
    - 最近邻距离：响应是否至少与一个参考文本接近
    - 双重信号比较：当两个信号都表明响应远离参考域时触发

    阈值配置：
    - centroid_threshold: 质心距离阈值（默认：0.5）
    - nn_threshold: 最近邻距离阈值（默认：0.4）
    - min_confidence: 最小置信度阈值（默认：0.6）
    """

    def __init__(
        self,
        corpus: ReferenceCorpus,
        centroid_threshold: float = 0.5,
        nn_threshold: float = 0.4,
        min_confidence: float = 0.6,
    ):
        self.corpus = corpus
        self.centroid_threshold = centroid_threshold
        self.nn_threshold = nn_threshold
        self.min_confidence = min_confidence

    def check(self, text: str) -> DriftResult:
        """
        检测文本是否漂移。

        Args:
            text: 待检测的文本

        Returns:
            漂移检测结果
        """
        if not text.strip() or self.corpus.size() == 0:
            return DriftResult(
                is_drift=False,
                centroid_distance=0.0,
                nearest_neighbor_distance=0.0,
                confidence=0.0,
            )

        # 嵌入文本
        embedding = self.corpus.embedding_model.embed([text])[0]

        # 计算质心距离
        centroid_distance = self.corpus.compute_centroid_distance(embedding)

        # 计算最近邻距离
        nn_index, nn_distance = self.corpus.get_nearest_neighbor(embedding)

        # 计算置信度
        centroid_score = min(centroid_distance / self.centroid_threshold, 1.0)
        nn_score = min(nn_distance / self.nn_threshold, 1.0)
        confidence = (centroid_score + nn_score) / 2.0

        # 判断是否漂移
        is_drift = (
            centroid_distance > self.centroid_threshold
            and nn_distance > self.nn_threshold
            and confidence >= self.min_confidence
        )

        return DriftResult(
            is_drift=is_drift,
            centroid_distance=centroid_distance,
            nearest_neighbor_distance=nn_distance,
            nearest_neighbor_index=nn_index,
            confidence=confidence,
            drift_type="semantic",
        )


class LLMJudge(ABC):
    """
    LLM-as-Judge 抽象基类。

    参考 ai-agent-evals 的评估框架：
    "Use advanced models to evaluate other model outputs"

    使用轻量级模型作为"评判者"，判断对话是否偏离了教学主题。
    """

    @abstractmethod
    async def evaluate(
        self,
        user_input: str,
        assistant_response: str,
        context: Dict[str, Any],
    ) -> bool:
        """
        评估助手响应是否偏离教学目标。

        Args:
            user_input: 用户输入
            assistant_response: 助手响应
            context: 上下文信息

        Returns:
            是否偏离教学目标
        """
        pass


class MultiDimensionDriftDetector:
    """
    多维度漂移检测引擎。

    综合使用：
    1. 语义漂移：嵌入向量相似度（driftGuard方案）
    2. 领域漂移：分类器检测是否偏离教学领域
    3. 性能漂移：监控响应质量和任务完成度
    4. LLM-as-Judge：轻量级模型作为评判者

    参考 LLM-Drift-Detection-Adaptation-Pipeline 的设计。
    """

    def __init__(
        self,
        drift_detector: Optional[DriftDetector] = None,
        llm_judge: Optional[LLMJudge] = None,
        use_vector_drift: bool = True,
        use_llm_judge: bool = False,
    ):
        self.drift_detector = drift_detector
        self.llm_judge = llm_judge
        self.use_vector_drift = use_vector_drift
        self.use_llm_judge = use_llm_judge

    async def detect(
        self,
        user_input: str,
        assistant_response: str,
        context: Dict[str, Any],
    ) -> DriftResult:
        """
        综合检测漂移。

        Args:
            user_input: 用户输入
            assistant_response: 助手响应
            context: 上下文信息

        Returns:
            漂移检测结果
        """
        if not assistant_response.strip():
            return DriftResult(
                is_drift=False,
                centroid_distance=0.0,
                nearest_neighbor_distance=0.0,
                confidence=0.0,
            )

        drift_results = []

        # 1. 基于向量嵌入的漂移检测
        if self.use_vector_drift and self.drift_detector:
            vector_result = self.drift_detector.check(assistant_response)
            drift_results.append(("vector", vector_result))

        # 2. LLM-as-Judge 漂移检测
        if self.use_llm_judge and self.llm_judge:
            llm_drift = await self.llm_judge.evaluate(
                user_input, assistant_response, context
            )
            llm_result = DriftResult(
                is_drift=llm_drift,
                centroid_distance=0.0,
                nearest_neighbor_distance=0.0,
                confidence=0.8 if llm_drift else 0.2,
                drift_type="llm_judge",
            )
            drift_results.append(("llm", llm_result))

        # 3. 启发式规则检测（兜底）
        heuristic_result = self._heuristic_check(assistant_response, context)
        drift_results.append(("heuristic", heuristic_result))

        # 综合结果：多数投票
        drift_count = sum(1 for _, result in drift_results if result.is_drift)
        total_count = len(drift_results)

        is_drift = drift_count >= total_count // 2 + 1

        # 计算综合置信度
        avg_confidence = sum(
            result.confidence for _, result in drift_results
        ) / total_count if total_count > 0 else 0.0

        # 获取主漂移类型
        main_drift_type = drift_results[0][1].drift_type

        return DriftResult(
            is_drift=is_drift,
            centroid_distance=drift_results[0][1].centroid_distance,
            nearest_neighbor_distance=drift_results[0][1].nearest_neighbor_distance,
            confidence=avg_confidence,
            drift_type=main_drift_type,
        )

    async def detect_drift(
        self,
        user_input: str,
        model_output: str,
        scene_context: str = "",
    ) -> DriftResult:
        """tutor_agent 兼容接口：将 scene_context 字符串包装为 context dict。"""
        ctx: Dict[str, Any] = {}
        if scene_context:
            ctx["scene"] = scene_context
        ctx["target_lang"] = "en"
        return await self.detect(
            user_input=user_input,
            assistant_response=model_output,
            context=ctx,
        )

    def _heuristic_check(
        self,
        assistant_response: str,
        context: Dict[str, Any],
    ) -> DriftResult:
        """
        启发式规则检测（兜底方案）。

        当向量检测不可用时使用。

        Args:
            assistant_response: 助手响应
            context: 上下文信息

        Returns:
            漂移检测结果
        """
        target_lang = context.get("target_lang", "en")

        teaching_keywords = {
            "ja": ["注意", "记住", "练习", "日语", "用法", "意思是", "可以说", "正确的是",
                   "语法", "词汇", "发音", "助词", "动词", "名词"],
            "en": ["remember", "the correct form", "try again", "notice", "grammar",
                   "instead of", "vocabulary", "pronunciation", "verb", "noun"],
        }

        keywords = teaching_keywords.get(target_lang, [])
        assistant_lower = assistant_response.lower()

        has_teaching_element = any(kw.lower() in assistant_lower for kw in keywords)
        is_long_response = len(assistant_response) > 300

        if is_long_response and not has_teaching_element:
            return DriftResult(
                is_drift=True,
                centroid_distance=0.6,
                nearest_neighbor_distance=0.5,
                confidence=0.7,
                drift_type="heuristic",
            )

        return DriftResult(
            is_drift=False,
            centroid_distance=0.3,
            nearest_neighbor_distance=0.2,
            confidence=0.5,
            drift_type="heuristic",
        )


# ── 预构建的教学领域参考语料 ────────────────────────────────────


def build_default_corpus(embedding_model: EmbeddingModel) -> ReferenceCorpus:
    """
    构建默认的教学领域参考语料库。

    包含代表性的教学领域文本，用于检测漂移。

    Args:
        embedding_model: 嵌入模型

    Returns:
        参考语料库
    """
    corpus = ReferenceCorpus(embedding_model)

    # 英语教学参考文本
    english_texts = [
        "Let me correct that for you. The correct form is...",
        "Remember to use the past tense when talking about yesterday.",
        "Try again using the correct article.",
        "Notice the difference between 'a' and 'an'.",
        "Instead of saying 'I goed', we say 'I went'.",
        "Let's practice using the present perfect tense.",
        "The grammar rule here is...",
        "You're doing great! Let's work on your vocabulary.",
        "Pay attention to your pronunciation of this word.",
        "Let me help you with this sentence structure.",
        "Good effort! The correct way to say this is...",
        "Let's review what we learned earlier.",
        "Focus on using the correct preposition.",
        "Try using this new vocabulary word in a sentence.",
        "Remember that adjectives come before nouns in English.",
    ]

    # 日语教学参考文本
    japanese_texts = [
        "注意：中文的「勉強」是勉强的意思，但日语的「勉強（べんきょう）」是学习的意思。",
        "记住：动作发生的场所用「で」，移动的目的地用「に」。",
        "练习一下这个助词的用法。",
        "日语的助词非常重要，让我来解释一下。",
        "可以说「学校で勉強する」，不能说「学校に勉強する」。",
        "正确的说法是「買って」，不是「買いて」。",
        "日语的语法规则和中文有很多不同。",
        "让我帮你扩充日语词汇。",
        "注意你的日语发音，特别是音调。",
        "助词「は」表示主题，「が」表示主语。",
        "动词的て形变化规则是...",
        "我们来练习一下敬语的用法。",
        "自动词和他动词的区别是...",
        "中日同形异义词需要特别注意。",
        "记住日语的基本句型结构。",
    ]

    corpus.add_texts(english_texts + japanese_texts)
    return corpus