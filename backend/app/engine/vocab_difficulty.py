# -*- coding: utf-8 -*-
"""用词难度审计：高频基础词表 + 超纲词统计（plans/0916-20 修复3）。

真机现象：用户等级 A2（校准卡在档），AI 却用 `settled / sorted / wrapped up /
casual way` 这类 B1–B2 词解释词义——等级此前只控"中英比例"，从不控"词有多难"。

这一版**只做审计**（只标记、不拦截、不改写回复）：解释轮统计回复里的超纲词，
超过阈值就打 `vocab_too_hard` 标记，供后续调优。词表遵循"先粗后细"：
这里是一份 A1/A2 最常用的基础词表（含全部功能词与常见变形），
宁可漏判也不要误判——它只进日志与审计，不影响用户看到的任何内容。
"""
import re
from typing import Iterable, List, Tuple

# A1/A2 基础词表（粗版）。含义：出现在这里的词算"最常见的基础词"，
# 不在表内、且不是数字/专名的实词才算超纲。改这份表要同步跑
# test_runner/test_vocab_difficulty.py，确认不把常见词判成超纲。
A2_CORE_WORDS = frozenset("""
a about above across act add afraid after afternoon again against age ago agree air all
allow almost alone along already also always am among an and angry animal another answer
any anybody anything anyway apartment appear apple are area arm around arrive art as ask at
ate away baby back bad bag ball bank bar base bath be beach bear beautiful because become
bed bedroom been before begin behind believe below best better between big bike bird birth
birthday bit black block blue board boat body book born both bottle box boy bread break
breakfast bring brother brown build building bus business busy but buy by bye call came can
cannot cap car card care carry case cat catch chair chance change cheap check chicken child
children choose city class clean clear climb clock close clothes cloud club coat coffee
cold collect college color come common company computer cook cool corner cost could country
course cover cross cry cup cut dad dance dark date daughter day dead dear decide deep
delicious did die different difficult dinner dirty do doctor does dog dollar done door
double down draw dress drink drive driver drop dry during each ear early east easy eat egg
eight either else email end enjoy enough enter even evening ever every everybody everyone
everything exact example exercise expect explain eye face fact fail fall family famous far
farm fast fat father favorite feel few field fight fill film find fine finger finish fire
first fish five fix flat floor flower fly follow food foot football for forget fork form
found four free fresh friend from front fruit full fun funny game garden get girl give
glad glass go goal god gold good goodbye got grandfather grandmother grass great green
grey ground group grow guess guy hair half hall hand happen happy hard has hat hate have he
head hear heart heavy hello help her here hers hey hi high him his history hit hold holiday
home hope horse hospital hot hotel hour house how hundred hungry hurry husband i ice idea
if ill important in inside instead interest internet into is it its job join joke journey
juice jump just keep key kid kill kind king kitchen know lady lake lamp land language large
last late later laugh learn least leave left leg less lesson let letter level library life
light like line listen little live local long look lose lot loud love low lunch machine
main make man many map market marry match matter may maybe me meal mean meat meet member
menu message middle might mile milk mind minute miss mistake mom moment monday money month
moon more morning most mother mountain mouth move movie much mum music must my name near
need never new news next nice night nine no nobody noise none noon north not note nothing
now number of off offer office often oh oil old on once one only open or orange order other
our out outside over own page paint pair paper parent park part party pass past pay pen
pencil people perhaps person phone photo pick picture piece pig place plan plane plant
plate play please point police pool poor popular possible post pound practice prefer
prepare present pretty price problem program put quarter question quick quiet quite radio
rain raise read ready real really reason receive red remember repeat reply report rest
return rice rich ride right river road rock room round rule run sad safe said salt same
sand save say school science sea season seat second see seem sell send sentence september
serious serve service set seven several she sheep shop short should shoulder show sick side
sign silver simple since sing sister sit six size skirt sky sleep slow small smart smell
smile snow so some somebody someone something sometimes son song soon sorry sound soup
south space speak special spell spend sport spring stand star start station stay step still
stop store story street strong student study stupid subject such sugar summer sun sunday
supermarket sure sweet swim table take talk tall taste taxi tea teach teacher team tell ten
terrible test than thank that the their them then there these they thin thing think third
thirsty this those though thought thousand three through throw thursday ticket tidy time
tired title to today together tomorrow tonight too tooth top touch town toy train travel
tree trip trouble true try tuesday turn twelve twenty twice two type ugly under understand
university until up us use usually vacation very video village visit voice wait wake walk
wall want war warm wash waste watch water way we wear weather wedding wednesday week
weekend well went were west wet what when where which while white who whole why wide wife
wild will win wind window wine winter with without woman wonder word work world worry would
write wrong year yellow yes yesterday yet you young your zero
""".split())

# 课堂高频词补充：这些词在"解释词义"的场景里几乎每轮都出现，漏了会把
# 正常解释判成超纲（粗版词表的常见坑：只收生活词、漏掉教学元词汇）。
_CLASSROOM_WORDS = frozenset("""
phrase meaning grammar translate correct wrong probably actually verb noun adjective adverb
pronounce sound letter express talk say tell ask answer question sentence example
""".split())

A2_CORE_WORDS = A2_CORE_WORDS | _CLASSROOM_WORDS

# 缩写/黏着部分与常见变形后缀：命中即算基础词，避免"you've / played"被误判超纲。
CLITIC_PARTS = frozenset({"ve", "re", "ll", "s", "t", "d", "m", "n"})
_SUFFIXES = ("ing", "ed", "es", "s", "er", "est", "ly", "n", "y")
# y 变 i 的变形（easier -> easy、happiest -> happy）：先还原再剥后缀。
_Y_RESTORE = {"ier": "y", "iest": "y"}
_TOKEN_RE = re.compile(r"[a-z]+")

# 审计阈值：解释轮里**不同**超纲词达到这个数就打标记（只标记，不拦截）。
# 定成 2 而不是 3：真机事故那一轮的回复里超纲词本来就只有 1-2 个
# （settled / dealing），阈值 3 会把事故本身漏掉。
HARD_WORD_MARK_THRESHOLD = 2


def normalize_token(token: str) -> str:
    """把常见变形还原成词表里的原形（粗版：剥后缀，够用即可）。"""
    word = token.strip().lower()
    if word in A2_CORE_WORDS or word in CLITIC_PARTS:
        return word
    for suffix, replacement in _Y_RESTORE.items():
        if word.endswith(suffix) and word[: -len(suffix)] + replacement in A2_CORE_WORDS:
            return word[: -len(suffix)] + replacement
    for suffix in _SUFFIXES:
        if len(word) > len(suffix) + 1 and word.endswith(suffix):
            base = word[: -len(suffix)]
            if len(base) >= 2 and base in A2_CORE_WORDS:
                return base
            # 双写末字母（stopped -> stop）
            if len(base) > 2 and base[-1] == base[-2] and base[:-1] in A2_CORE_WORDS:
                return base[:-1]
    return word


def hard_words(text: str, allowed: Iterable[str] = A2_CORE_WORDS) -> List[str]:
    """回复里不是最基础词的实词（去重、按首次出现顺序）。

    全大写的 token（COD、CS、NBA…）视为缩写/专名**不计超纲**——难度审计关心的是
    普通词汇，而缩写另有一条专门的本盘术语锚机制（plans/0916-20 修复2）。
    """
    raw = str(text or "")
    allowed_set = set(allowed)
    acronyms = {token.lower() for token in re.findall(r"\b[A-Z]{2,}\b", raw)}
    seen: List[str] = []
    for token in _TOKEN_RE.findall(raw.lower()):
        if len(token) <= 1 or token in acronyms:
            continue
        normalized = normalize_token(token)
        if normalized in allowed_set or normalized in CLITIC_PARTS:
            continue
        if normalized not in seen:
            seen.append(normalized)
    return seen


def audit_vocab_difficulty(text: str) -> Tuple[int, List[str]]:
    """返回 ``(不同超纲词数, 超纲词列表)``。"""
    words = hard_words(text)
    return len(words), words
