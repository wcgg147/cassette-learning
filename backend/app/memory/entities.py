"""Small deterministic guards for extracted entity rows."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import re
from typing import Any


_ALLOWED_ENTITY_TYPES = {"person", "pet", "place", "event", "thing", "other"}
_GENERIC_PERSON_LABELS = {
    "friend", "朋友", "同学", "同事", "亲戚", "家人", "同伴", "老师", "外教",
    "friend [1]", "friend [2]", "朋友 [1]", "朋友 [2]",
}
_ENTITY_SEARCH_STOPWORDS = {
    "a", "about", "after", "again", "also", "an", "at", "because", "before", "could", "from", "in",
    "have", "just", "know", "like", "more", "other", "really", "that", "them",
    "then", "there", "they", "this", "those", "what", "when", "where", "which",
    "the", "with", "would", "you", "your", "my", "friend", "friends", "朋友", "同学", "同事", "家人", "用户",
    "我们", "你们", "他们", "她们", "这个", "那个", "事情", "问题", "感觉",
    "知道", "怎么", "什么", "为什么", "还有", "好像", "就是", "一个", "现在",
    "今天", "昨天", "之前", "刚才", "因为", "所以", "但是", "然后", "自己",
}


def stable_entity_id(user_id: str, etype: str, name: str, conv_id: str) -> str:
    """Return an opaque, conversation-scoped ID unless a known ID was matched.

    A display name is not proof of identity: two people may share a name. The
    archive path can explicitly reuse a known ID when evidence supports the
    match; otherwise this ID remains stable only for rewrites of this tape.
    """
    clean_type = str(etype or "other").strip().lower()
    clean_name = " ".join(str(name or "").split()).casefold()
    base = f"{str(user_id)}\x1f{clean_type}\x1f{clean_name}\x1f{str(conv_id)}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def entity_search_terms(text: str, max_terms: int = 48) -> list[str]:
    """Build bounded lexical probes for cross-session entity lookup.

    This is a retrieval hint, not an entity resolver. Exact identity decisions
    still require explicit archive evidence; ambiguous results are kept separate.
    """
    raw = str(text or "").casefold()
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        term = " ".join(str(term or "").split()).strip()
        if (len(term) < 2 or term in _ENTITY_SEARCH_STOPWORDS or term in seen
                or term.isdigit()):
            return
        seen.add(term)
        terms.append(term)

    for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", raw):
        if re.fullmatch(r"[a-z0-9]+", token):
            if len(token) >= 3:
                add(token)
            continue
        if 2 <= len(token) <= 16:
            add(token)
        for index in range(len(token) - 1):
            add(token[index:index + 2])

    if len(terms) <= max_terms:
        return terms
    # Keep probes from both the beginning and end; users often add the actual
    # question/name after a long spoken preamble.
    head = max_terms // 2
    return terms[:head] + terms[-(max_terms - head):]


def entity_match_score(query: str, entity: dict) -> int:
    """Rank an entity against one utterance without merging identities."""
    query_text = " ".join(str(query or "").casefold().split())
    name = " ".join(str((entity or {}).get("name") or "").casefold().split())
    detail = " ".join(str((entity or {}).get("detail") or "").casefold().split())
    generic = name in _GENERIC_PERSON_LABELS
    query_terms = set(entity_search_terms(query, max_terms=96))
    name_terms = (
        set() if generic else
        set(entity_search_terms(name, max_terms=32)) - _ENTITY_SEARCH_STOPWORDS
    )
    detail_terms = set(entity_search_terms(detail, max_terms=48)) - _ENTITY_SEARCH_STOPWORDS
    name_hits = len(query_terms & name_terms)
    detail_hits = len(query_terms & detail_terms)
    if name and not generic and name in query_text:
        # Keep exact names dominant while allowing a specific detail such as a
        # season or event clue to select its matching observation.
        return 100 + detail_hits
    if name_hits:
        return 20 + 5 * name_hits + detail_hits
    # One distinctive detail is enough to retrieve a candidate. Retrieval may
    # surface more than one person; the separate archive identity gate remains
    # stricter and must never merge on this score alone. Generic role words are
    # excluded above so "friend" by itself cannot pull every friend into prompt.
    if detail_hits >= 1:
        return 5 + detail_hits
    return 0


def merge_entity_search_observations(
    rows: list[dict], query: str, limit: int = 12, max_observations: int = 3,
) -> list[dict]:
    """Return one ranked entity row with a small, dated evidence trail.

    Entity IDs can have observations in several conversations. Returning only
    the newest row made older but still relevant details invisible on a direct
    per-turn lookup. Keep the representative row compatible with current
    callers, and attach a bounded list of distinct observations for prompt
    assembly. This is retrieval aggregation; it does not merge stored rows.
    """
    safe_limit = max(1, min(int(limit), 24))
    observation_cap = max(1, min(int(max_observations), 6))
    by_id: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for raw in rows or []:
        if not isinstance(raw, dict):
            continue
        entity_id = str(raw.get("entity_id") or "").strip()
        if not entity_id:
            continue
        score = entity_match_score(query, raw)
        if score > 0:
            by_id[entity_id].append((score, raw))

    ranked = []
    for entity_id, observations in by_id.items():
        observations.sort(
            key=lambda item: (
                item[0],
                str(item[1].get("observed_at") or item[1].get("created_at") or ""),
            ),
            reverse=True,
        )
        best_score, representative = observations[0]
        merged = dict(representative)
        representative_name = " ".join(
            str(representative.get("name") or "").casefold().split()
        )
        query_text = " ".join(str(query or "").casefold().split())
        # When the user names a non-generic entity, they are asking about that
        # entity's timeline, not only the one observation with the highest
        # lexical score. Keep its bounded trail; fuzzy detail-only hits remain
        # score-filtered so a loose clue cannot pull unrelated history.
        include_entity_timeline = bool(
            representative_name
            and representative_name not in _GENERIC_PERSON_LABELS
            and representative_name in query_text
        )
        detail_observations = []
        seen_details: set[str] = set()
        for score, row in observations:
            if score != best_score and not include_entity_timeline:
                continue
            detail = " ".join(str(row.get("detail") or "").split()).strip()
            signature = detail.casefold()
            if not detail or signature in seen_details:
                continue
            seen_details.add(signature)
            detail_observations.append({
                "detail": detail[:300],
                "observed_at": str(row.get("observed_at") or row.get("created_at") or "")[:48],
                "evidence_source": str(row.get("evidence_source") or "unknown"),
                "confidence": row.get("confidence"),
                "evidence_ref": str(row.get("evidence_ref") or "")[:64],
            })
            if len(detail_observations) >= observation_cap:
                break
        merged["detail_observations"] = detail_observations
        ranked.append((best_score, entity_id, merged))

    ranked.sort(
        key=lambda item: (
            item[0],
            str(item[2].get("observed_at") or item[2].get("created_at") or ""),
            item[1],
        ),
        reverse=True,
    )
    return [entity for _score, _entity_id, entity in ranked[:safe_limit]]


def prepare_entity_relation_rows(
    user_id: str, conv_id: str, relations: list[dict]
) -> list[tuple]:
    """Normalize append-only entity edges; never store source transcript text."""
    rows: list[tuple] = []
    seen: set[str] = set()
    for relation in relations or []:
        if not isinstance(relation, dict):
            continue
        subject_id = str(relation.get("subject_entity_id") or "").strip().lower()
        object_id = str(relation.get("object_entity_id") or "").strip().lower()
        predicate = str(relation.get("predicate") or "").strip().lower()
        object_value = " ".join(str(relation.get("object_value") or "").split())[:240]
        if not re.fullmatch(r"[0-9a-f]{32}", subject_id):
            continue
        if object_id and not re.fullmatch(r"[0-9a-f]{32}", object_id):
            continue
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", predicate):
            continue
        if not object_id and not object_value:
            continue
        evidence_source = str(relation.get("evidence_source") or "unknown")
        if evidence_source not in {"raw_asr", "stored_user_message_text", "unknown"}:
            evidence_source = "unknown"
        evidence_ref = str(relation.get("evidence_ref") or "")[:64]
        evidence_quote = " ".join(str(relation.get("evidence_quote") or "").split())[:160]
        try:
            confidence = float(relation.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))
        try:
            source_message_seq = int(relation.get("source_message_seq"))
        except (TypeError, ValueError):
            source_message_seq = None
        signature = "\x1f".join((subject_id, predicate, object_id, object_value, evidence_ref))
        edge_key = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:32]
        if edge_key in seen:
            continue
        seen.add(edge_key)
        rows.append((user_id, str(conv_id), edge_key, subject_id, predicate,
                     object_id, object_value, evidence_source, evidence_ref, confidence,
                     source_message_seq, evidence_quote))
    return rows


def normalize_archive_entity_graph(
    user_id: str,
    conv_id: str,
    raw_entities: list[dict],
    raw_relations: list[dict],
    messages: list,
    known_entities: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Validate archive claims against user-message evidence and attach IDs.

    Raw ASR is the only acceptable evidence when it exists for a message. The
    polished/display transcript is a fallback source only when no raw ASR was
    supplied. Quotes are hashed into evidence_ref; transcript text itself is
    never copied to the entity tables. A known ID is reused only when the
    evidence uniquely supports it; ambiguous matches remain separate.
    """
    known_by_id: dict[str, dict] = {}
    known_ids_by_name: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in known_entities or []:
        if not isinstance(item, dict):
            continue
        entity_id = str(item.get("entity_id") or "").strip().lower()
        if re.fullmatch(r"[0-9a-f]{32}", entity_id):
            known_by_id.setdefault(entity_id, item)
            known_ids_by_name[(
                str(item.get("etype") or "other").lower(),
                " ".join(str(item.get("name") or "").casefold().split()),
            )].add(entity_id)

    def supports_known_identity(known: dict, etype: str, quote: str) -> bool:
        if str(known.get("etype") or "other").lower() != etype:
            return False
        known_id = str(known.get("entity_id") or "").strip().lower()
        known_name = " ".join(str(known.get("name") or "").casefold().split())
        quote_folded = " ".join(str(quote or "").casefold().split())
        name_is_generic = known_name in _GENERIC_PERSON_LABELS
        exact_name = bool(
            known_name and not name_is_generic and known_name in quote_folded
            and len(known_ids_by_name.get((etype, known_name), set())) <= 1
        )
        quote_terms = set(entity_search_terms(quote, max_terms=96))
        detail_tokens = set(entity_search_terms(
            str(known.get("detail") or ""), max_terms=64
        )) - _ENTITY_SEARCH_STOPWORDS
        supported_by_details = len(quote_terms & detail_tokens) >= 2
        # A compound descriptor can be sufficient only when it contains a
        # distinctive word and belongs to exactly one known identity.
        detail_words = re.findall(
            r"[a-z0-9]+|[\u4e00-\u9fff]+",
            " ".join(str(known.get("detail") or "").casefold().split()),
        )
        generic_heads = {
            "friend", "friends", "person", "people", "朋友", "同学", "同事",
            "亲戚", "家人", "同伴", "老师", "外教",
        }
        supported_by_compound_descriptor = any(
            f"{left} {right}" in quote_folded
            and any(token not in generic_heads | _ENTITY_SEARCH_STOPWORDS
                    for token in (left, right))
            for left, right in zip(detail_words, detail_words[1:])
        )
        return bool(known_id and (
            exact_name or supported_by_details or supported_by_compound_descriptor
        ))

    def field(message: Any, name: str, default: Any = "") -> Any:
        if isinstance(message, dict):
            return message.get(name, default)
        return getattr(message, name, default)

    def evidence_for(claim: dict) -> tuple[str, str, int, str] | None:
        try:
            index = int(claim.get("evidence_message")) - 1
        except (TypeError, ValueError):
            return None
        quote = " ".join(str(claim.get("evidence_quote") or "").split())
        if not quote or len(quote) > 160 or index < 0 or index >= len(messages):
            return None
        message = messages[index]
        if str(field(message, "role", "")) != "user":
            return None
        raw = " ".join(str(field(message, "raw_content", "") or "").split())
        content = " ".join(str(field(message, "content", "") or "").split())
        if raw:
            if quote not in raw:
                return None
            source = "raw_asr"
        else:
            if quote not in content:
                return None
            source = "stored_user_message_text"
        evidence_ref = hashlib.sha256(
            f"{user_id}\x1f{conv_id}\x1f{index + 1}\x1f{quote}".encode("utf-8")
        ).hexdigest()
        return source, evidence_ref, index, quote

    candidates = []
    seen = set()
    for raw in raw_entities or []:
        if not isinstance(raw, dict):
            continue
        etype = str(raw.get("etype") or "other").strip().lower()
        name = " ".join(str(raw.get("name") or "").split())[:120]
        detail = " ".join(str(raw.get("detail") or "").split())[:300]
        ref = str(raw.get("ref") or "").strip()[:40]
        if not name or etype not in _ALLOWED_ENTITY_TYPES or not ref:
            continue
        evidence = evidence_for(raw)
        if evidence is None:
            continue
        source, evidence_ref, source_message_seq, evidence_quote = evidence
        signature = (ref, etype, name.casefold(), detail.casefold(), evidence_ref)
        if signature in seen:
            continue
        seen.add(signature)
        requested_id = str(raw.get("existing_entity_id") or "").strip().lower()
        existing_id = requested_id
        known = known_by_id.get(existing_id)
        if requested_id:
            # A provider-selected ID is only a candidate. Reject it unless the
            # user's quoted evidence supports that exact known identity.
            if not known or not supports_known_identity(known, etype, evidence_quote):
                existing_id = ""
                known = None
        else:
            # Extraction can miss the optional ID field. Recover only when the
            # quoted evidence matches exactly one known identity; ambiguity and
            # bare generic labels continue to create a tape-local node.
            supported_ids = [
                entity_id for entity_id, candidate in known_by_id.items()
                if supports_known_identity(candidate, etype, evidence_quote)
            ]
            if len(supported_ids) == 1:
                existing_id = supported_ids[0]
                known = known_by_id[existing_id]
            else:
                # 2026-10-25 内容兜底匹配（实体层大修·治"一事多行"）：
                # 精确身份匹配只认 quote 支撑 known 的 name/detail——归档器对
                # 同一事件起不同名（"朋友来访/接待朋友"）、同一人不同措辞时
                # 全部落空，各自新开行（真机 Jason 案例：一次会话的事被存成
                # 7 行，模型据此把"计划带朋友爬山"断言成"用户常爬山"，反事
                # 实实验已实锤归因，见 PROGRESS 实体层大修档）。兜底判据＝
                # bigram 独有内容词 <3（与渲染层取证同判据）：detail 相对
                # 某 known 实体只差"他/其"级措辞 ⇒ 同一事件的另一版本 ⇒
                # 复用该实体行（save_entities 的 upsert 会把 detail 并入）。
                # 边界：仅 real 人/事；scene 戏内实体保留原隔离语义；
                # 泛称（friend/朋友）不并——两位朋友会被错拼成一人。
                if (
                    etype in ("person", "event")
                    and detail
                    and len(supported_ids) == 0
                ):
                    from memory.entity_tokens import content_token_set as _cts
                    new_tokens = _cts(detail)
                    if new_tokens:
                        merged_hits = []
                        for cand_id, cand in known_by_id.items():
                            if str(cand.get("etype") or "other").lower() not in ("person", "event"):
                                continue
                            cand_name = " ".join(str(cand.get("name") or "").casefold().split())
                            if cand_name in _GENERIC_PERSON_LABELS:
                                continue
                            cand_detail = str(cand.get("detail") or "")
                            cand_tokens = _cts(cand_detail)
                            if not cand_tokens:
                                continue
                            if len(new_tokens - cand_tokens) < 3 and len(cand_tokens - new_tokens) < 3:
                                merged_hits.append(cand_id)
                        if len(merged_hits) == 1:
                            existing_id = merged_hits[0]
                            known = known_by_id[existing_id]
                        else:
                            existing_id = ""
                            known = None
                    else:
                        existing_id = ""
                        known = None
                else:
                    existing_id = ""
                    known = None
        if known:
            # A matched identity uses its established display name. Do not let
            # a generic alias silently rename the node during archive writes.
            name = " ".join(str(known.get("name") or name).split())[:120]
        try:
            confidence = float(raw.get("confidence", 0.65))
        except (TypeError, ValueError):
            confidence = 0.65
        candidate = {
            "ref": ref,
            "etype": etype,
            "name": name,
            "detail": detail,
            "entity_id": existing_id,
            "evidence_source": source,
            "evidence_ref": evidence_ref,
            "source_message_seq": source_message_seq,
            "evidence_quote": evidence_quote,
            "confidence": min(1.0, max(0.0, confidence)),
        }
        candidates.append(candidate)

    # Separate same-label, different-detail mentions deterministically. If the
    # extractor linked both to one known ID, they are one identity and merge.
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in candidates:
        groups[(item["etype"], item["name"].casefold())].append(item)
    entities: list[dict] = []
    ref_to_id: dict[str, str] = {}
    for (_, _), group in groups.items():
        distinct_ids = {item["entity_id"] for item in group if item["entity_id"]}
        distinct_details = {item["detail"].casefold() for item in group}
        same_known_identity = (
            len(distinct_ids) == 1
            and all(item["entity_id"] in distinct_ids for item in group)
        )
        collide = len(distinct_details) > 1 and not same_known_identity
        ordered = sorted(group, key=lambda item: (item["detail"].casefold(), item["ref"]))
        for index, item in enumerate(ordered, start=1):
            name = f"{item['name']} [{index}]" if collide else item["name"]
            entity_id = item["entity_id"] or stable_entity_id(
                user_id, item["etype"], name, str(conv_id)
            )
            item["name"] = name
            item["entity_id"] = entity_id
            ref_to_id[item["ref"]] = entity_id
            entities.append({key: value for key, value in item.items() if key != "ref"})

    relations: list[dict] = []
    seen_edges: set[tuple] = set()
    for raw in raw_relations or []:
        if not isinstance(raw, dict):
            continue
        subject_ref = str(raw.get("subject_ref") or "").strip()[:40]
        object_ref = str(raw.get("object_ref") or "").strip()[:40]
        subject_id = ref_to_id.get(subject_ref, "")
        object_id = ref_to_id.get(object_ref, "") if object_ref else ""
        predicate = str(raw.get("predicate") or "").strip().lower()
        object_value = " ".join(str(raw.get("object_value") or "").split())[:240]
        if not subject_id or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", predicate):
            continue
        if object_ref and not object_id:
            continue
        if not object_id and not object_value:
            continue
        evidence = evidence_for(raw)
        if evidence is None:
            continue
        source, evidence_ref, source_message_seq, evidence_quote = evidence
        try:
            confidence = float(raw.get("confidence", 0.65))
        except (TypeError, ValueError):
            confidence = 0.65
        signature = (subject_id, predicate, object_id, object_value, evidence_ref)
        if signature in seen_edges:
            continue
        seen_edges.add(signature)
        relations.append({
            "subject_entity_id": subject_id,
            "predicate": predicate,
            "object_entity_id": object_id,
            "object_value": object_value,
            "evidence_source": source,
            "evidence_ref": evidence_ref,
            "confidence": min(1.0, max(0.0, confidence)),
            "source_message_seq": source_message_seq,
            "evidence_quote": evidence_quote,
        })
    return entities, relations


def entity_neighborhood_ids(
    seed_ids: set[str], relation_rows: list[dict], limit: int = 4,
) -> list[str]:
    """Return a small, evidence-backed one-hop neighborhood for matched nodes.

    Direct text matches stay authoritative. This only adds entities connected
    by a stored relation with usable confidence; it never infers a new edge or
    merges two IDs.
    """
    seeds = {str(value or "").strip() for value in seed_ids if str(value or "").strip()}
    cap = max(0, min(int(limit), 8))
    if cap <= 0 or not seeds:
        return []
    related: list[str] = []
    seen = set(seeds)
    for row in relation_rows or []:
        if not isinstance(row, dict):
            continue
        try:
            confidence = float(row.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < 0.45:
            continue
        subject_id = str(row.get("subject_entity_id") or "").strip()
        object_id = str(row.get("object_entity_id") or "").strip()
        if subject_id not in seeds and object_id not in seeds:
            continue
        for entity_id in (subject_id, object_id):
            if entity_id and entity_id not in seen:
                seen.add(entity_id)
                related.append(entity_id)
                if len(related) >= cap:
                    return related
    return related


def assemble_entity_neighborhood(
    seed_entities: list[dict],
    neighbor_entities: list[dict],
    relation_rows: list[dict],
    *,
    neighbor_limit: int = 4,
) -> list[dict]:
    """Attach stored edges and a capped one-hop neighbor set to direct matches."""
    result: list[dict] = []
    by_id: dict[str, dict] = {}
    for raw in seed_entities or []:
        if not isinstance(raw, dict):
            continue
        entity_id = str(raw.get("entity_id") or "").strip()
        if not entity_id or entity_id in by_id:
            continue
        entity = dict(raw)
        entity["relations"] = []
        by_id[entity_id] = entity
        result.append(entity)

    allowed_neighbors = set(entity_neighborhood_ids(
        set(by_id), relation_rows, limit=neighbor_limit,
    ))
    for raw in neighbor_entities or []:
        if not isinstance(raw, dict):
            continue
        entity_id = str(raw.get("entity_id") or "").strip()
        if not entity_id or entity_id not in allowed_neighbors or entity_id in by_id:
            continue
        entity = dict(raw)
        entity["relations"] = []
        by_id[entity_id] = entity
        result.append(entity)

    for raw in relation_rows or []:
        if not isinstance(raw, dict):
            continue
        subject_id = str(raw.get("subject_entity_id") or "").strip()
        subject = by_id.get(subject_id)
        if not subject:
            continue
        relation = dict(raw)
        relation.pop("subject_entity_id", None)
        object_id = str(relation.get("object_entity_id") or "").strip()
        if not str(relation.get("object_name") or "").strip() and object_id in by_id:
            relation["object_name"] = str(by_id[object_id].get("name") or "")
        subject["relations"].append(relation)
    return result


def prepare_entity_rows(user_id: str, conv_id: str, entities: list[dict]) -> list[tuple]:
    """Return idempotent storage rows without collapsing same-label entities.

    The legacy storage key includes ``name`` and ``conv_id``. If an extractor
    emits two distinct people as ``friend`` in one conversation, a normal
    upsert would silently replace one detail with the other. Preserve both by
    assigning deterministic, conversation-local suffixes only to colliding
    labels. Identical duplicate claims still collapse to one row. This is a
    collision guard, not a claim that the suffix is a cross-conversation ID.
    """
    candidates: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    metadata_by_signature: dict[tuple[str, str, str], dict] = {}
    for entity in entities or []:
        if not isinstance(entity, dict):
            continue
        etype = str(entity.get("etype") or "other").strip().lower()
        name = str(entity.get("name") or "").strip()[:120]
        if not name or etype not in _ALLOWED_ENTITY_TYPES:
            continue
        detail = str(entity.get("detail") or "")[:300]
        signature = (etype, name.casefold(), " ".join(detail.split()).casefold())
        if signature in seen:
            continue
        seen.add(signature)
        metadata_by_signature[signature] = entity
        candidates.append((etype, name, detail))

    groups: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for candidate in candidates:
        groups[(candidate[0], candidate[1].casefold())].append(candidate)

    rows: list[tuple] = []
    for (etype, _), group in groups.items():
        # Order by evidence text, not extractor order, so rerunning the same
        # archive with reordered JSON does not swap the local suffixes.
        ordered = sorted(group, key=lambda row: (" ".join(row[2].split()).casefold(), row[1]))
        explicit_ids = {
            str(metadata_by_signature[(etype, name.casefold(), " ".join(detail.split()).casefold())]
                .get("entity_id") or "").strip().lower()
            for _, name, detail in ordered
        }
        same_known_identity = len(explicit_ids) == 1 and all(explicit_ids)
        collides = len(ordered) > 1 and not same_known_identity
        for index, (_, name, detail) in enumerate(ordered, start=1):
            stored_name = f"{name} [{index}]" if collides else name
            entity_id = stable_entity_id(user_id, etype, stored_name, str(conv_id))
            signature = (etype, name.casefold(), " ".join(detail.split()).casefold())
            metadata = metadata_by_signature[signature]
            supplied_id = "".join(str(metadata.get("entity_id") or "").split())
            if re.fullmatch(r"[0-9a-fA-F]{32}", supplied_id):
                entity_id = supplied_id.lower()
            evidence_source = str(metadata.get("evidence_source") or "stored_user_message_text")
            if evidence_source not in {"raw_asr", "stored_user_message_text", "unknown"}:
                evidence_source = "unknown"
            try:
                confidence = float(metadata.get("confidence", 0.65))
            except (TypeError, ValueError):
                confidence = 0.65
            confidence = min(1.0, max(0.0, confidence))
            evidence_ref = str(metadata.get("evidence_ref") or "")[:64]
            evidence_quote = " ".join(str(metadata.get("evidence_quote") or "").split())[:160]
            try:
                source_message_seq = int(metadata.get("source_message_seq"))
            except (TypeError, ValueError):
                source_message_seq = None
            rows.append((user_id, str(conv_id), etype, stored_name, detail,
                         entity_id, evidence_source, confidence, evidence_ref,
                         source_message_seq, evidence_quote))
    return rows
