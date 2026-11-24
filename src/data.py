"""Dataset loaders for MS MARCO passage retrieval and BEIR TREC-COVID."""
from typing import Dict, List, Set, Tuple
import random
from datasets import load_dataset


_MSMARCO_CACHE = {}


def _get_msmarco_validation():
    if "val" not in _MSMARCO_CACHE:
        _MSMARCO_CACHE["val"] = load_dataset("microsoft/ms_marco", "v2.1", split="validation")
    return _MSMARCO_CACHE["val"]


def load_msmarco_background(num_passages: int = 50000) -> List[str]:
    """Load unique passages from MS MARCO validation to compute offline corpus-wide IDF."""
    ds = _get_msmarco_validation()
    passages: List[str] = []
    for row in ds:
        for txt in row["passages"]["passage_text"]:
            if txt.strip():
                passages.append(txt)
            if len(passages) >= num_passages:
                return passages
    return passages


def build_msmarco_subset(
    seed: int = 0,
    num_eval_queries: int = 100,
    num_passages: int = 1000,
) -> Tuple[List[str], List[str], List[Set[int]]]:
    """Build a reproducible MS MARCO mini-corpus with guaranteed official positive passages."""
    ds = _get_msmarco_validation().shuffle(seed=seed)

    queries: List[str] = []
    query_gold_texts: List[List[str]] = []
    hard_neg_texts: List[str] = []
    filler_texts: List[str] = []

    scan_limit = min(len(ds), max(num_eval_queries * 10, num_passages // 2 + 500))
    for i in range(scan_limit):
        row = ds[i]
        passages = row["passages"]["passage_text"]
        selected = row["passages"]["is_selected"]
        pos = [p for p, s in zip(passages, selected) if s == 1 and p.strip()]
        neg = [p for p, s in zip(passages, selected) if s == 0 and p.strip()]

        if len(queries) < num_eval_queries and pos:
            queries.append(row["query"])
            query_gold_texts.append(pos)
            hard_neg_texts.extend(neg)
        else:
            filler_texts.extend([p for p in passages if p.strip()])

        if len(queries) >= num_eval_queries and (len(hard_neg_texts) + len(filler_texts)) >= num_passages * 2:
            break

    corpus: List[str] = []
    text_to_idx: Dict[str, int] = {}

    def add_passage(txt: str) -> int:
        if txt not in text_to_idx:
            text_to_idx[txt] = len(corpus)
            corpus.append(txt)
        return text_to_idx[txt]

    gold_sets: List[Set[int]] = []
    for pos_list in query_gold_texts:
        gold_sets.append({add_passage(p) for p in pos_list})

    for txt in hard_neg_texts:
        if len(corpus) >= num_passages:
            break
        add_passage(txt)

    for txt in filler_texts:
        if len(corpus) >= num_passages:
            break
        add_passage(txt)

    return queries, corpus, gold_sets


def build_trec_covid_subset(
    num_passages: int = 5000,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[Dict[int, int]]]:
    """Load all 50 official BEIR TREC-COVID queries with graded relevance judgments."""
    q_ds = load_dataset("BeIR/trec-covid", "queries", split="queries")
    qrels_ds = load_dataset("BeIR/trec-covid-qrels", split="test")
    corpus_ds = load_dataset("BeIR/trec-covid", "corpus", split="corpus")

    qid_to_rel: Dict[str, Dict[str, int]] = {}
    needed_doc_ids: Set[str] = set()
    for row in qrels_ds:
        score = int(row["score"])
        if score > 0:
            qid = str(row["query-id"])
            did = str(row["corpus-id"])
            qid_to_rel.setdefault(qid, {})[did] = score
            needed_doc_ids.add(did)

    rng = random.Random(seed)
    all_corpus_rows = {}
    distractors = []
    for row in corpus_ds:
        did = str(row["_id"])
        txt = (row.get("title", "") + " " + row.get("text", "")).strip()
        if not txt:
            continue
        if did in needed_doc_ids:
            all_corpus_rows[did] = txt
        elif len(distractors) < num_passages:
            distractors.append((did, txt))

    rng.shuffle(distractors)
    selected_doc_ids: List[str] = []
    seen: Set[str] = set()
    if len(all_corpus_rows) <= num_passages:
        for did in all_corpus_rows:
            selected_doc_ids.append(did)
            seen.add(did)
    else:
        for qid, rel_map in qid_to_rel.items():
            ranked_dids = sorted(rel_map.keys(), key=lambda d: rel_map[d], reverse=True)
            for did in ranked_dids[: max(10, num_passages // (2 * len(qid_to_rel)))]:
                if did in all_corpus_rows and did not in seen:
                    selected_doc_ids.append(did)
                    seen.add(did)
        for did in all_corpus_rows:
            if len(selected_doc_ids) >= int(num_passages * 0.65):
                break
            if did not in seen:
                selected_doc_ids.append(did)
                seen.add(did)

    corpus: List[str] = []
    did_to_idx: Dict[str, int] = {}
    for did in selected_doc_ids:
        did_to_idx[did] = len(corpus)
        corpus.append(all_corpus_rows[did])

    for did, txt in distractors:
        if len(corpus) >= num_passages:
            break
        if did not in did_to_idx:
            did_to_idx[did] = len(corpus)
            corpus.append(txt)

    queries: List[str] = []
    qrels_list: List[Dict[int, int]] = []
    for row in q_ds:
        qid = str(row["_id"])
        rel_map = qid_to_rel.get(qid, {})
        mapped = {did_to_idx[d]: s for d, s in rel_map.items() if d in did_to_idx}
        if mapped:
            queries.append(row["text"])
            qrels_list.append(mapped)

    return queries, corpus, qrels_list
