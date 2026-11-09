from enum import Enum
from dataclasses import dataclass
import re
from typing import Optional


class TutorIntent(Enum):
    GREETING = "greeting"
    FAREWELL = "farewell"
    EXPRESSION_SUPPORT = "expression_support"
    QUESTION = "question"
    PRACTICE = "practice"
    CHITCHAT = "chitchat"
    COMPLAINT = "complaint"
    CONFUSION = "confusion"
    OFF_TOPIC = "off_topic"
    META_QUESTION = "meta_question"
    SCENARIO_ROLEPLAY = "scenario_roleplay"


HIGH_EMOTION_WORDS = [
    "根本不听", "自说自话", "没感觉", "变态", "好烦", "卡住了",
    "烦透了", "受不了", "气死了", "滚", "闭嘴", "讨厌死了",
]


@dataclass
class IntentResult:
    intent: TutorIntent
    confidence: float
    matched_rule: str
    emotion_intensity: str = "low"


class IntentRouter:
    """Rule-based intent router. No LLM calls. <1ms latency."""

    RULES = [
        (TutorIntent.GREETING, [
            r"^(嗨|哈喽|你好|hello|hi|hey|morning|evening|afternoon)",
            r"(我又来了|早上好|下午好|晚上好|good\s*(morning|evening|night))",
        ], 0.9),
        (TutorIntent.FAREWELL, [
            r"(?:再见|拜拜|晚安|不聊了|先这样|(?:我|那我|我们)?\s*(?:先)?(?:走了|去睡(?:觉)?了|休息了)|"
            r"(?:今天|那|我们)?(?:就)?聊到(?:这里|这儿))",
            r"\b(?:bye(?:\s+for\s+now)?|goodbye|good\s*night|see\s+you(?:\s+(?:later|soon|tomorrow))?|"
            r"talk\s+to\s+you\s+(?:later|soon)|catch\s+you\s+(?:later|soon)|gotta\s+go|have\s+to\s+go)\b|"
            r"^\s*(?:later|night)[.!?]*$",
        ], 0.9),
        # Open-world scenario negotiation must win over the broad word "练习".
        # Roles are deliberately not enumerated here.
        (TutorIntent.SCENARIO_ROLEPLAY, [
            r"(角色扮演|演绎|情景.{0,8}(练习|训练|对话)|场景.{0,8}(练习|训练|对话))",
            r"(把我当成|你(?:就)?当成|你来当|我来当|扮演|作为).{1,24}",
            r"(?:和|跟).{1,18}(?:怎么|如何)?(?:沟通|交流|对话|说话)",
            r"(试试|来一个|练习一下|来一遍|过一遍).{0,16}(场景|情景|角色)",
        ], 0.95),
        (TutorIntent.META_QUESTION, [
            r"为什么(要|不|一定|得|必须|偏|总是)",
            r"你的逻辑",
            r"你怎么想的",
            r"你这样做是",
        ], 0.95),
        # Repair intent must win over broad complaint/question patterns such as
        # "怎么" and "什么意思".
        (TutorIntent.CONFUSION, [
            r"(不懂|不明白|什么意思|不理解|不知道怎么说|怎么说来着|不知道)",
            r"(i\s*don'?t\s*(understand|know)|what\s*do\s*you\s*mean|huh\??)",
        ], 0.92),
        (TutorIntent.EXPRESSION_SUPPORT, [
            r"(?:怎么|如何|怎样).{0,18}(?:寻求帮助|求助|要回|找回|开口|表达|处理|解决)",
            r"(?:用|换成)?(?:英文|英语|english).{0,8}(?:怎么|如何|怎样)?(?:说|表达|讲|写)",
            r"(?:寻求帮助|怎么求助|用英文|用英语|英语怎么说|英文怎么说|怎么跟.{1,8}(?:说|讲|开口))",
            r"(?:落在|丢在|忘在|找不到了|不见了).{0,25}(?:怎么|想|要|英文|英语|求助)",
            r"(?:想|要).{0,16}(?:把.{1,10}(?:拿回来|找回|要回)|联系|打电话|找司机|找客服)",
        ], 0.9),
        (TutorIntent.COMPLAINT, [
            r"(太烦|好烦|烦死了|讨厌|又是这样|怎么又|为什么又|为什么每次|为什么突然)",
            r"(为什么每次都|你不了解|你不记得|你忘了|你变了|跟之前说的不一样)",
            r"(你.{0,12}(根本不听|没听我|不听我|没有针对|没针对|没有贴合|没贴合)|"
            r"(没有针对|没针对).{0,12}(输入|问题|需求|我说的))",
            r"(第三方.{0,8}(评估|评价)|评估角色|记忆.{0,8}(错乱|错了|混乱)|"
            r"好像.{0,8}(第三方|评估))",
            r"((这个|当前|刚才的)?话题.{0,6}(太)?无聊)",
            # Natural product feedback is often phrased as a description of
            # the failure rather than with an explicit emotion word.
            r"(答非所问|偏题|跑题|离题|脱离.{0,12}(话题|问题|上文)|"
            r"(回复|回答|问题|话题).{0,16}(没.{0,4}关系|没有.{0,6}关系|无关|脱离))",
            r"((没.{0,4}关系|没有.{0,6}关系|无关).{0,16}(回复|回答|问题|话题)|"
            r"不是已经.{0,20}(说过|练过|用过)|又回到|重复.{0,12}(句式|练习))",
            r"(突然.{0,8}(全英文|纯英文|只用英文)|为什么突然.{0,12}(全英文|纯英文|只用英文))",
        ], 0.8),
        (TutorIntent.QUESTION, [
            r"[？?]$",
            r"(聊了啥|说什么了|聊了什么|说了什么|什么意思)",
            r"^(为什么|怎么|什么|哪里|谁|多少|几|how|what|why|where|when|who|is|are|do|does|can|could|would|should|will)\b",
            r"(啥时候|何时|几点|哪儿|哪些|哪个|几岁|多大|多远|多长)",
            r"(天气|气温|温度).{0,10}(怎么样|如何|多少度|几度)",
        ], 0.85),
        (TutorIntent.OFF_TOPIC, [
            r"(讲个笑话|聊点别的|换个话题|不聊这个|政治|宗教|股票|彩票)",
        ], 0.7),
    ]

    def __init__(self):
        self._compiled = {}
        for intent, patterns, confidence in self.RULES:
            self._compiled[intent] = [
                (re.compile(p, re.IGNORECASE), confidence) for p in patterns
            ]

    def classify(self, user_message: str) -> IntentResult:
        text = user_message.strip().lower()

        for intent, rules in self._compiled.items():
            for pattern, confidence in rules:
                if pattern.search(text):
                    emotion_intensity = self._detect_emotion_intensity(text, intent)
                    return IntentResult(
                        intent=intent,
                        confidence=confidence,
                        matched_rule=pattern.pattern,
                        emotion_intensity=emotion_intensity,
                    )

        emotion_intensity = self._detect_emotion_intensity(text, TutorIntent.CHITCHAT)
        return IntentResult(
            intent=TutorIntent.CHITCHAT,
            confidence=0.5,
            matched_rule="default",
            emotion_intensity=emotion_intensity,
        )

    def _detect_emotion_intensity(self, text: str, intent: TutorIntent) -> str:
        for word in HIGH_EMOTION_WORDS:
            if word in text:
                return "high"
        if intent == TutorIntent.META_QUESTION:
            if re.search(r"((?<!不)很(?!多)|(?<!不)太(?!多)|非常|真的|实在|超级|到底)", text):
                return "high"
            return "medium"
        if intent in (TutorIntent.COMPLAINT, TutorIntent.CONFUSION):
            # "英语不太好" and "我不太懂" describe proficiency or local
            # difficulty; the old ``太`` pattern promoted them to high
            # frustration and incorrectly selected a complaint-repair move.
            if re.search(r"((?<!不)很(?!多)|(?<!不)太(?!多)|非常|真的|实在|超级)", text):
                return "high"
            return "medium"
        return "low"
