# RAG Dataset Format Guide: Legal RAG Bench Style

A practitioner's guide to building a custom RAG evaluation dataset in the **Legal RAG Bench** format (Isaacus, 2026). Written defensively — every design decision is traceable to the Hugging Face dataset card, the Isaacus release blog, or the arXiv paper, with the author's own additions clearly labelled.

---

## What this format is, and why you'd copy it

Legal RAG Bench is an **end-to-end** RAG evaluation set. The defining feature, and the reason to imitate it specifically rather than e.g. LegalBench-RAG, is that every question is labelled with **both** the gold passage **and** a hand-written long-form answer. That dual labelling is what enables Isaacus's contribution: a hierarchical error-decomposition framework that attributes each end-to-end failure to one of hallucination, retrieval, or reasoning. (See §6 for the methodology — the format only makes sense once you understand the methodology it serves.)

DeepEval, RAGAS, and similar evaluation libraries can be retrofitted to consume this format, but the format is not designed *for* them. It is designed for Isaacus's three-binary-metric error decomposition. Treat library compatibility as a downstream convenience, not a design rationale.

Sources consulted: Hugging Face dataset card `isaacus/legal-rag-bench`; Isaacus blog *Introducing Legal RAG Bench* (20 Feb 2026); arXiv 2603.01710 (Butler & Butler, 2 Mar 2026).

---

## Table of contents

1. [Format overview](#1-format-overview)
2. [Corpus format](#2-corpus-format)
3. [QA format](#3-qa-format)
4. [Question authoring discipline](#4-question-authoring-discipline)
5. [Validation](#5-validation)
6. [Evaluation methodology](#6-evaluation-methodology)
7. [Publishing to Hugging Face](#7-publishing-to-hugging-face)
8. [Integration with eval-harness](#8-integration-with-eval-harness)
9. [Common pitfalls](#9-common-pitfalls)
10. [Worked example](#10-worked-example)
11. [Adversarial review checklist](#11-adversarial-review-checklist)
12. [Quick reference](#12-quick-reference)

---

## 1. Format overview

Legal RAG Bench uses **two configurations** (Hugging Face calls them "subsets") of the same dataset, each with a single `test` split:

| Config | Purpose | Rows in original | Key linking field |
|--------|---------|------------------|-------------------|
| `corpus` | Source passages indexed for retrieval | 4,876 | `id` (referenced by qa) |
| `qa` | Hand-written questions, answers, gold-passage labels | 100 | `relevant_passage_id` (foreign key into corpus) |

There is **no train/val split.** Legal RAG Bench is evaluation-only. If your colleague proposes splits, push back: splits imply training use the benchmark is not designed for.

---

## 2. Corpus format

### 2.1 Directory layout

```
your-dataset/
├── README.md            # Dataset card with YAML config
├── corpus/
│   └── test.jsonl       # One passage per line
└── qa/
    └── test.jsonl       # One question per line
```

JSONL is easier to diff in version control than monolithic JSON. The Hugging Face `datasets` library loads both natively.

### 2.2 Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | **Yes** | Unique, stable, hierarchy-encoding identifier. Must be referenced verbatim by `qa.relevant_passage_id`. |
| `title` | string | **Yes** | Section heading from the source document. Present on the live dataset even though the README sometimes omits it — confirm against the data files, not just the README. |
| `text` | string | **Yes** | Passage content in Markdown. Capped at ≤512 tokens under the Kanon tokenizer in the original. |
| `footnotes` | string \| null | **Yes** (nullable) | Markdown footnote definitions referenced from `text`. Use `null`, not `""`, when absent — this matches the source dataset and avoids `datasets`-library schema-inference issues. |

### 2.3 Sample row (verbatim from the live dataset)

```json
{
  "id": "1.1-c1-s1",
  "title": "1.1 Introductory Remarks",
  "text": "# 1.1 Introductory Remarks\n\n1. A number of studies into the jury system have suggested that it is highly beneficial for the judge to provide the jury with information at the beginning of a trial...",
  "footnotes": "[^1]: See, e.g. Parliament of Victoria Law Reform Committee, *Jury Service in Victoria*, Final Report (1991)..."
}
```

### 2.4 The ID convention matters

Isaacus's IDs encode source hierarchy: `1.2-c2-s3` reads as *section 1.2, chunk 2, sub-chunk 3*. Preserve this style because:

- A human reading the ID can locate the passage in the source without joining back to anything.
- Gaps in `s` numbering expose dropped chunks during pipeline development.
- It is type-stable (always a string), unlike auto-incrementing integers, and remains valid if you re-chunk.

#### ID examples by domain

| Domain | Example IDs |
|--------|-------------|
| HR policy | `hr_2024_3_2_c1_s1`, `hr_2024_3_2_c1_s2` |
| Master service agreement | `msa_2024_cl8_1_c1_s1`, `msa_2024_cl8_1_c2_s1` |
| API documentation | `api_v2_users_create_c1_s1`, `api_v2_auth_oauth2_c1_s1` |
| Financial regulation | `basel3_ccr_rwa_c1_s1`, `basel3_mar_sa_c1_s1` |

#### ID anti-patterns

```jsonl
{"id": "sha256_a3f5b9c1d2e8...", ...}  // Hash-based: breaks on content change
{"id": 1, ...}                          // Integer: no semantic meaning; type mismatches with qa.relevant_passage_id (string)
{"id": "550e8400-e29b-41d4-a716-...", ...}  // UUID: impossible to debug
```

### 2.5 Chunking

Isaacus's pipeline:

1. Convert source documents (Word `.docx` in their case) to Markdown, preserving headings, lists, emphasis, and footnotes.
2. Apply heuristics to break sections into their hierarchy (chapters, subchapters, sections). Isaacus does not open-source those heuristics — write your own that respect your source's structure.
3. Where a hierarchical chunk still exceeds the token budget, further chunk with **`semchunk`** (`github.com/isaacus-dev/semchunk`) targeting **≤512 tokens under the Kanon tokenizer** (`huggingface.co/isaacus/kanon-tokenizer`).

The 512-token cap matters. It's tight enough that retrievers can't game the benchmark by retrieving giant blobs. If you change the cap or the tokenizer, document the change and don't claim parity with Legal RAG Bench's reported results.

**If you don't want the Kanon tokenizer dependency**, pin an alternative explicitly. `tiktoken` with `cl100k_base` is a common default. Word counts (e.g. "200–500 words") are **not** a substitute — words ≠ tokens, and word-based limits are unreproducible across implementations.

### 2.6 Markdown formatting

- Headings (`#`, `##`) preserved.
- Numbered lists, emphasis preserved.
- Footnotes use Markdown footnote syntax (`[^1]`) in `text`, with definitions in `footnotes`.

Because footnote references in `text` only resolve via `footnotes`, your inference-time prompt template must concatenate both fields when presenting a passage to a retriever or an LLM. Stripping `footnotes` will silently lose citations.

### 2.7 What does **not** go in the corpus

- Stub entries (e.g. a heading with no body). Filter these out during chunking. A passage that has no text content cannot serve as ground truth.
- Pure navigation/boilerplate (table-of-contents, "Welcome to...", copyright notices). These pollute retrieval and are never the gold passage for any question.
- Duplicates. Two passages with the same text but different IDs will confuse retrieval evaluation — a retriever that finds one is "right" by the gold label but indistinguishable from a retriever that finds the other.

---

## 3. QA format

### 3.1 Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | **string** | Yes | Unique question identifier. **String, not integer** — matches the source dataset's type and stays type-consistent with `relevant_passage_id`. |
| `question` | string | Yes | Hand-written, expert-crafted, deliberately lexically dissimilar from the gold passage. |
| `answer` | string | Yes | Hand-written long-form answer (typically 1–3 sentences). Must be derivable from the gold passage alone. Not a label, not a span, not yes/no. |
| `relevant_passage_id` | string | Yes | A single `corpus.id` value. **Not a list.** See §3.3 for why. |

### 3.2 Sample row (paraphrased from the worked example in the Isaacus blog)

```json
{
  "id": "q-042",
  "question": "Sally is accused of cultivating narcotic plants in her backyard. One of the elements of this charge is that 'the accused intentionally cultivated or attempted to cultivate a particular substance.' To establish whether this is the case, the judge believes it would be valuable to visit Sally's backyard and have the jury examine it for themselves. What is the name of the legal procedure whereby the court travels to a location relevant to the charge?",
  "answer": "A view. Under Evidence Act 2008 (Vic) s 53 the court may order a 'demonstration, experiment or inspection' (collectively, a 'view'), and an inspection involves the court travelling to view a location.",
  "relevant_passage_id": "2.1-c1-s1"
}
```

Note: the question's surface vocabulary is about narcotics cultivation; the answer lives in the evidence chapter under "views." This is the lexical-dissimilarity property in action (see §4.2).

### 3.3 Why a single `relevant_passage_id`, not a list

The schema is a string, not a list, by deliberate design:

> *"passages were randomly sampled to produce 100 handwritten, complex, and meaningfully challenging questions that, to the maximum extent possible, would require **each of those passages alone** to be answered correctly."* — HF dataset card

The question author's job is to ensure exactly one passage suffices. If you want to label multiple gold passages, you have either:

- An under-specified question — rewrite it to target a single passage, or
- A corpus with unintended duplication — merge passages during chunking, or
- A multi-hop retrieval task — that's a different evaluation paradigm with different metrics; don't mix paradigms by quietly changing the schema.

If you genuinely need multi-passage support, fork the format and document the fork. Don't pretend it's still Legal RAG Bench.

### 3.4 Answer writing

- **Self-contained.** Bad: `"See section 3.2."` Good: `"Employees must submit leave requests at least 48 hours in advance through the HR portal."`
- **Uses corpus terminology.** If the passage says "associates," don't switch to "employees" in the answer.
- **Long-form.** Not one-word, not yes/no, not a multiple-choice letter. The point of the long-form answer is to allow an LLM-as-judge to score entailment between a generated response and a reference response, which only works if the reference is substantive.
- **Derivable from the gold passage alone.** If you need facts the passage doesn't contain, rewrite the question or pick a different passage.

---

## 4. Question authoring discipline

This is where most replications fail. The format is the easy part; the questions are the hard part.

### 4.1 Sampling-then-writing, not the other way round

Random-sample passages from your corpus **first**, then hand-write a question targeting each sampled passage. Do **not** brainstorm questions and look for matching passages — that produces lexically similar, surface-level questions any embedder solves trivially. Do **not** ask an LLM to generate questions from passages — same failure mode, plus the LLM's biases.

The sampling-first order forces the human author to engage with the passage on its merits rather than working from a question they already had in mind.

### 4.2 Lexical dissimilarity is a hard requirement, not a guideline

> *"In drafting questions, we made them as lexically dissimilar from relevant passages as possible in order to stress test the semantic understanding of evaluated models."* — Isaacus blog

Concrete rules:

- Do not reuse distinctive nouns, verbs, or phrases from the passage in the question.
- Frame the question as a real-world scenario or hypothetical, not a fact lookup.
- Do not reference the passage's section number, heading, or position in the source.

**Verify, don't just intend.** Build a BM25 index over your corpus, run each question through it, and check the rank of the gold passage. A defensible target: the gold passage should **not** appear in the top 5 BM25 results for the question. If it does, the question is too lexically aligned with the passage — rewrite it. (Isaacus does not publish a specific BM25-rank threshold; the >5 figure is a practical heuristic. Calibrate to your corpus size.)

```python
# Sketch: lexical-dissimilarity check
from rank_bm25 import BM25Okapi

corpus_ids = [doc["id"] for doc in corpus]
tokenized_corpus = [doc["text"].lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

failures = []
for qa in qa_records:
    scores = bm25.get_scores(qa["question"].lower().split())
    ranked_ids = [corpus_ids[i] for i in scores.argsort()[::-1]]
    gold_rank = ranked_ids.index(qa["relevant_passage_id"]) + 1
    if gold_rank <= 5:
        failures.append((qa["id"], gold_rank))

if failures:
    print(f"{len(failures)} questions too lexically similar to gold passage:")
    for qid, rank in failures:
        print(f"  {qid}: gold at BM25 rank {rank}")
```

### 4.3 Expert-knowledge bar

Each question should require expert-level domain knowledge. If a layperson with the passage in front of them can answer the question, you're testing reading comprehension, not RAG-under-expert-knowledge. Both are legitimate eval types, but they are different things — be honest about which yours is.

For Isaacus's Charge Book questions, "expert-level" means familiarity with Victorian criminal procedure. For your domain, define the equivalent bar explicitly in your annotation guide.

### 4.4 Scenario diversity

100 questions is small. Each must earn its place. Track coverage across:

- **Topics / chapters** in your corpus (no clustering in a single area).
- **Question types**: factual lookup, procedural, definitional, comparative, applied-scenario.
- **Difficulty levels**: some answerable by careful reading, some requiring genuine synthesis.

Keep a coverage matrix during annotation. Reviewers will ask for it.

### 4.5 Inter-annotator agreement

Have a second annotator independently re-label `relevant_passage_id` on a random 10% sample. Target ≥90% agreement. If agreement is lower, your annotation guidelines are too vague — fix the guidelines and re-annotate, don't ship.

This is the cheapest credibility insurance against an adversarial reviewer asking "how do you know your labels are right?"

---

## 5. Validation

### 5.1 Pre-publication validation script

```python
"""
Validation for Legal-RAG-Bench-style datasets.

Checks:
- Structural integrity (JSON validity, required fields, type consistency)
- Referential integrity (qa.relevant_passage_id resolves into corpus)
- Distributional sanity (no over-concentration of gold passages)
- Lexical-dissimilarity (BM25 rank of gold passage given question)
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
    HAVE_BM25 = True
except ImportError:
    HAVE_BM25 = False


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_corpus(corpus: list[dict]) -> list[str]:
    errors = []
    seen_ids = set()
    for i, doc in enumerate(corpus, 1):
        for field in ("id", "title", "text"):
            if field not in doc:
                errors.append(f"corpus[{i}]: missing field '{field}'")
        if "footnotes" not in doc:
            errors.append(f"corpus[{i}]: missing field 'footnotes' (use null if absent)")
        if not isinstance(doc.get("id"), str):
            errors.append(f"corpus[{i}]: 'id' must be string, got {type(doc.get('id')).__name__}")
        if not doc.get("text", "").strip():
            errors.append(f"corpus[{i}] id={doc.get('id')!r}: empty 'text'")
        if doc.get("id") in seen_ids:
            errors.append(f"corpus[{i}]: duplicate id {doc['id']!r}")
        seen_ids.add(doc.get("id"))
    return errors


def validate_qa(qa: list[dict], corpus_ids: set[str]) -> list[str]:
    errors = []
    seen_ids = set()
    for i, q in enumerate(qa, 1):
        for field in ("id", "question", "answer", "relevant_passage_id"):
            if field not in q:
                errors.append(f"qa[{i}]: missing field '{field}'")
        if not isinstance(q.get("id"), str):
            errors.append(f"qa[{i}]: 'id' must be string (matches source schema)")
        if not isinstance(q.get("relevant_passage_id"), str):
            errors.append(f"qa[{i}]: 'relevant_passage_id' must be string, not list — "
                          "Legal RAG Bench uses single-passage gold labels")
        pid = q.get("relevant_passage_id")
        if pid and pid not in corpus_ids:
            errors.append(f"qa[{i}] id={q.get('id')!r}: relevant_passage_id {pid!r} not in corpus")
        if q.get("id") in seen_ids:
            errors.append(f"qa[{i}]: duplicate id {q['id']!r}")
        seen_ids.add(q.get("id"))
    return errors


def report_distribution(qa: list[dict], corpus_ids: set[str]) -> None:
    usage = Counter(q["relevant_passage_id"] for q in qa if q.get("relevant_passage_id"))
    used = set(usage)
    unused = corpus_ids - used
    print(f"\n=== DISTRIBUTION ===")
    print(f"Corpus passages: {len(corpus_ids)}")
    print(f"QA questions: {len(qa)}")
    print(f"Passages used as gold: {len(used)} ({len(used)/len(corpus_ids):.1%})")
    print(f"Passages never used: {len(unused)}")
    dist = Counter(usage.values())
    print("Questions per passage:")
    for k in sorted(dist):
        print(f"  {k} question(s): {dist[k]} passages")


def check_lexical_dissimilarity(
    qa: list[dict], corpus: list[dict], top_n_threshold: int = 5
) -> list[tuple[str, int]]:
    """
    Return (qa_id, gold_rank) for questions where the gold passage
    appears too high in BM25 results (rank <= top_n_threshold).

    Heuristic, author's addition. Isaacus does not publish a specific
    threshold; calibrate to your corpus size and the difficulty you want.
    """
    if not HAVE_BM25:
        print("\n[skipped] rank_bm25 not installed; pip install rank_bm25")
        return []

    corpus_ids = [doc["id"] for doc in corpus]
    bm25 = BM25Okapi([doc["text"].lower().split() for doc in corpus])

    failures = []
    for q in qa:
        scores = bm25.get_scores(q["question"].lower().split())
        ranked = [corpus_ids[i] for i in scores.argsort()[::-1]]
        try:
            rank = ranked.index(q["relevant_passage_id"]) + 1
        except ValueError:
            continue
        if rank <= top_n_threshold:
            failures.append((q["id"], rank))
    return failures


def validate(corpus_path: Path, qa_path: Path) -> bool:
    corpus = load_jsonl(corpus_path)
    qa = load_jsonl(qa_path)
    corpus_ids = {doc["id"] for doc in corpus if "id" in doc}

    corpus_errors = validate_corpus(corpus)
    qa_errors = validate_qa(qa, corpus_ids)
    all_errors = corpus_errors + qa_errors

    report_distribution(qa, corpus_ids)

    failures = check_lexical_dissimilarity(qa, corpus, top_n_threshold=5)
    if failures:
        print(f"\n=== LEXICAL-DISSIMILARITY WARNINGS ({len(failures)}) ===")
        print("Gold passage appears in BM25 top 5 — question may be too easy:")
        for qid, rank in failures[:10]:
            print(f"  qa id={qid!r}: gold at BM25 rank {rank}")

    if all_errors:
        print(f"\n=== ERRORS ({len(all_errors)}) ===")
        for err in all_errors[:30]:
            print(f"  ✗ {err}")
        return False
    print("\n✓ Structural validation passed.")
    return True


if __name__ == "__main__":
    validate(Path("corpus/test.jsonl"), Path("qa/test.jsonl"))
```

### 5.2 Manual validation checklist

**Corpus**
- [ ] Every passage has a unique, hierarchy-encoding string `id`.
- [ ] No passage exceeds the documented token cap under the documented tokenizer.
- [ ] No passage is a stub (heading-only, boilerplate, navigation).
- [ ] No duplicate passage text.
- [ ] Markdown renders correctly when round-tripped.
- [ ] Footnote references in `text` all resolve in `footnotes`.
- [ ] `footnotes` is `null` (not `""`) when absent.

**QA**
- [ ] Every `relevant_passage_id` resolves into the corpus.
- [ ] `id` and `relevant_passage_id` are both strings.
- [ ] No duplicate `id`.
- [ ] Coverage table shows reasonable spread across topics and question types.
- [ ] Second annotator agreed on `relevant_passage_id` for ≥90% of a 10% sample.
- [ ] BM25 lexical-dissimilarity check passes (gold passage not in top-5 BM25 hits) — *author's heuristic, not from the source*.

**Answers**
- [ ] Every answer is derivable from the gold passage alone.
- [ ] No answer is one-word or yes/no.
- [ ] Answers use the corpus's terminology.

---

## 6. Evaluation methodology

This is the contribution that makes Legal RAG Bench distinctive. The format only earns its complexity once you run the methodology against it.

### 6.1 Three binary outcomes per (question, embedder, LLM)

For each question *i*, embedder *e*, LLM *l*, judge three binary outcomes:

- **Correctness** *c<sub>eli</sub>* — does the LLM's response entail the gold `answer`? (1/0)
- **Groundedness** *g<sub>eli</sub>* — is the response supported by whatever passages were retrieved, irrespective of whether those passages are actually the gold? (1/0)
- **Retrieval accuracy** *r<sub>eli</sub>* — was the gold passage retrieved? (1/0)

Isaacus uses GPT-5.2 in high-reasoning mode as the LLM-as-judge. Pin your judge model and your judging prompt and publish both — judge choice is a real confound.

### 6.2 Error decomposition

Apply this taxonomy **in order**:

1. **Hallucination** — `g = 0`. Response not grounded in retrieved passages, regardless of correctness. Checked first because, in legal practice, an ungrounded "correct" answer is indistinguishable from an ungrounded incorrect one — both are unverifiable.
2. **Retrieval error** — `g = 1 ∧ c = 0 ∧ r = 0`. Grounded, wrong, gold not retrieved. Attributable to the embedder.
3. **Reasoning error** — `g = 1 ∧ c = 0 ∧ r = 1`. Grounded, wrong, gold was retrieved. Attributable to the LLM.

The order matters. A correct answer that ignores the (irrelevant) retrieved context is still a hallucination under this framework, because in production a user cannot verify it.

### 6.3 Full factorial design

Evaluate **every** embedder × LLM combination. Isaacus evaluated 3 embedders × 2 LLMs = 6 combinations × 100 questions = 600 runs. Hold pipeline hyperparameters fixed across runs (Isaacus used a barebones LangChain RAG, temperature 0, otherwise defaults) so embedder and LLM are the only variables.

### 6.4 Reproducibility artefacts

For an adversarial reviewer to trust your numbers, release:

- The RAG pipeline code (or a precise spec: chunker, retriever top-k, prompt template, temperature, max tokens).
- Judge prompts, verbatim.
- Per-question raw outputs from every model combination.
- Retrieved passage IDs at each rank for every question.
- LLM-as-judge verdicts on every question, with the judge's reasoning if available.

Isaacus ships an interactive data explorer with all of this. That level of transparency is what blocks the "you cherry-picked" objection.

---

## 7. Publishing to Hugging Face

### 7.1 Directory structure

```
your-dataset/
├── README.md
├── LICENSE
├── corpus/
│   └── test.jsonl
└── qa/
    └── test.jsonl
```

### 7.2 README YAML

```markdown
---
dataset_info:
  - config_name: corpus
    data_files:
      - split: test
        path: corpus/*.jsonl
    features:
      - name: id
        dtype: string
      - name: title
        dtype: string
      - name: text
        dtype: string
      - name: footnotes
        dtype: string
  - config_name: qa
    data_files:
      - split: test
        path: qa/*.jsonl
    features:
      - name: id
        dtype: string
      - name: question
        dtype: string
      - name: answer
        dtype: string
      - name: relevant_passage_id
        dtype: string
task_categories:
  - text-retrieval
  - question-answering
language:
  - en
license: cc-by-nc-sa-4.0
---
```

Note: `id` is `string` in **both** configs. The qa `id` must be a string to stay type-consistent with `relevant_passage_id`, which is itself a string because it's a foreign key into corpus `id`. Integer `id`s will cause type-validation failures downstream.

### 7.3 Upload

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload your-org/your-dataset . --repo-type dataset
```

### 7.4 Licence

CC BY-NC-SA 4.0 mirrors Isaacus's licence. Confirm that:

1. Your source documents permit your release licence. If the source is more restrictive, that restriction cascades.
2. You attribute the source corpus.
3. You add the citations required by your source (for Legal RAG Bench imitations, cite the Legal RAG Bench paper and MLEB — see §12).

---

## 8. Integration with eval-harness

### 8.1 Dataset loader

File: `src/eval_harness/datasets/your_dataset.py`

```python
"""
Custom RAG dataset loader, following load_legal_rag_bench() pattern.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Final, Optional

from beartype import beartype

DATASET_NAME: Final[str] = "your-org/your-dataset"
DEFAULT_CACHE_DIR: Final[Path] = Path("data/rag/your_dataset/")

SLICE_PICO: Final[int] = 2
SLICE_NANO: Final[int] = 10


@beartype
def _get_slice_limit(slice_name: str) -> Optional[int]:
    limits = {"pico": SLICE_PICO, "nano": SLICE_NANO, "full": None}
    if slice_name not in limits:
        raise ValueError(f"slice must be 'pico', 'nano', or 'full', got: {slice_name}")
    return limits[slice_name]


@beartype
def load_your_dataset(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    slice: str = "full",
    force_refresh: bool = False,
) -> Iterator[tuple[str, str, str, str]]:
    """
    Yields: (query_id, query_text, relevant_passage_id, reference_answer)
    """
    from datasets import load_dataset

    limit = _get_slice_limit(slice)
    dataset = load_dataset(
        DATASET_NAME,
        name="qa",
        split="test",
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    for count, item in enumerate(dataset):
        yield (
            str(item["id"]),
            item["question"],
            item["relevant_passage_id"],
            item["answer"],
        )
        if limit is not None and count + 1 >= limit:
            break
```

### 8.2 Register

```python
# src/eval_harness/datasets/__init__.py
from eval_harness.datasets.your_dataset import load_your_dataset

__all__ = [
    "load_legalbench_rag",
    "load_omnidocbench",
    "load_dp_bench",
    "load_your_dataset",
]
```

---

## 9. Common pitfalls

**1. Type mismatch on `id`.** Source schema is `string` for both `corpus.id` and `qa.id`. Using `int` for qa `id` breaks downstream code that compares it against `relevant_passage_id`.

**2. List instead of string for `relevant_passage_id`.** The schema is a single string. If you want multi-passage labels, that's a different format — fork the schema and document the fork.

**3. Case-sensitive ID mismatch.** `Doc_001` in corpus, `doc_001` in qa: silent failure. Prefer lowercase-with-underscores throughout.

**4. Mixing `""` and `null` for `footnotes`.** The source uses `null`. Mixing will break `datasets` schema inference. Pick one.

**5. Answer references facts not in the passage.** If your reference answer includes information the gold passage doesn't contain, the LLM-as-judge will mark grounded responses as incorrect. Either expand the passage or trim the answer.

**6. Wrong config name on load.** `load_dataset("your-org/your-dataset")` may default to the first config. Use `name="qa"` or `name="corpus"` explicitly.

**7. Wrong split.** Only `test` exists. `split="train"` will fail.

**8. Questions generated by an LLM from passages.** Produces lexically aligned questions any embedder solves. Hand-write, sample passages first.

**9. Stubs in the corpus.** Heading-only passages, "Welcome to..." pages, copyright notices. Filter during chunking.

**10. Word-count caps instead of token caps.** Unreproducible. Pin a tokenizer.

---

## 10. Worked example

Small, domain: company policy handbook. Showing the *shape* of the format. **Caveat**: a real Legal-RAG-Bench-style dataset for this domain would require expert-knowledge questions (an HR specialist could pose questions involving statutory interaction, jurisdictional variance, or precedent — things a layperson can't answer even with the passage). The example below is illustrative of the schema, not of the expert-knowledge bar in §4.3.

### 10.1 Corpus (excerpt)

```jsonl
{"id": "handbook_values_c1_s1", "title": "Our Values", "text": "Our company operates on three core values:\n\n1. **Integrity**: We act honestly and ethically\n2. **Excellence**: We deliver quality work\n3. **Respect**: We treat everyone with dignity", "footnotes": null}
{"id": "handbook_hours_c1_s1", "title": "Working Hours", "text": "## Standard Hours\n\nBusiness hours are 9:00 AM to 5:00 PM, Monday through Friday.\n\n## Core Hours\n\nAll employees must be available 11:00 AM to 3:00 PM for meetings and collaboration.\n\n## Flexible Hours\n\nWith manager approval, start times may vary between 7-10 AM, but 8 hours must be worked.", "footnotes": null}
{"id": "handbook_remote_c1_s1", "title": "Remote Work", "text": "## Eligibility\n\nRemote work is available for employees whose roles can be performed off-site.\n\n## Requirements\n\n- Reliable internet connection\n- Dedicated workspace\n- Available during core hours\n- Manager approval required", "footnotes": null}
{"id": "handbook_leave_types_c1_s1", "title": "Types of Leave", "text": "## Annual Leave\n\n20 days per year, accruing monthly at 1.67 days.\n\n## Sick Leave\n\n10 days per year for personal illness or family care.\n\n## Personal Leave\n\n5 days per year for personal matters not covered by other leave types.\n\n## Bereavement Leave\n\n3 days paid leave for immediate family member death.", "footnotes": null}
{"id": "handbook_leave_process_c1_s1", "title": "Requesting Leave", "text": "## Notice Period\n\n- Annual leave: 2 weeks notice for >3 days\n- Sick/personal leave: By 9 AM on sick day\n- Bereavement: Notify when able\n\n## Approval\n\nSubmit requests via HR portal. Manager approval required for leaves >3 days.\n\n## Documentation\n\nMedical certificate required for sick leave >3 consecutive days.", "footnotes": null}
{"id": "handbook_expenses_c1_s1", "title": "Expense Reimbursement", "text": "## Allowable Expenses\n\n- Business travel (flights, hotels, meals)\n- Client entertainment (with prior approval)\n- Home office equipment (up to $500/year)\n\n## Submission\n\nSubmit within 30 days of expense via expense portal.\n\n## Approval\n\nManager approval required for expenses >$200.", "footnotes": null}
{"id": "handbook_conflict_c1_s1", "title": "Conflict of Interest", "text": "Employees must disclose any personal or financial interests that could conflict with company duties. This includes:\n\n- Outside employment\n- Family member employment at competitor\n- Personal investments in competitors or suppliers\n\nFailure to disclose may result in termination.", "footnotes": null}
{"id": "handbook_conduct_c1_s1", "title": "Code of Conduct", "text": "## Expected Behavior\n\n- Professional communication at all times\n- Respect for colleagues and clients\n- No harassment or discrimination\n- Protection of company confidential information\n\n## Reporting\n\nViolations should be reported to HR or via anonymous hotline. Retaliation against reporters is prohibited.", "footnotes": null}
```

Note no "Welcome" stub passage — that would fail the corpus rules in §2.7.

### 10.2 QA (excerpt)

```jsonl
{"id": "q-001", "question": "An employee discovers their spouse has just been hired as a senior buyer at a key supplier. They have not personally interacted with that supplier in their role, but their department occasionally awards contracts to it. What action does the handbook require?", "answer": "The employee must disclose the relationship. The conflict of interest policy requires disclosure of family member employment at suppliers, and failure to disclose may result in termination.", "relevant_passage_id": "handbook_conflict_c1_s1"}
{"id": "q-002", "question": "Chen wants to take a fortnight off in three weeks to attend a family wedding overseas. He plans to email his manager today. Will his request meet the policy's timing requirement?", "answer": "Yes. For annual leave longer than three days, two weeks' notice is required. Chen is providing three weeks' notice, which exceeds the requirement. He will also need manager approval because the leave is longer than three days.", "relevant_passage_id": "handbook_leave_process_c1_s1"}
{"id": "q-003", "question": "Fatima usually does her job from her flat. She has been asked to attend a planning workshop next Tuesday from 2 PM to 4 PM. Does the company's flexible-arrangements policy permit her to remain off-site for this session?", "answer": "No. Remote workers must be available during core hours, which are 11 AM to 3 PM. The 2-4 PM session overlaps with core hours, during which she must be available for meetings and collaboration — though the policy does not specifically require physical presence.", "relevant_passage_id": "handbook_remote_c1_s1"}
{"id": "q-004", "question": "Greg has been off work for four consecutive days with a stomach bug. When he returns, what does the company require from him beyond the standard sick-leave request?", "answer": "Greg must provide a medical certificate. Medical certificates are required for sick leave longer than three consecutive days.", "relevant_passage_id": "handbook_leave_process_c1_s1"}
{"id": "q-005", "question": "Aisha is setting up a permanent home workspace and wants to claim back the cost of an adjustable desk priced at $600. What will the company actually reimburse, and what additional step (if any) does she need to take?", "answer": "The company will reimburse up to $500 per year for home office equipment, so Aisha will receive $500 and must cover $100 herself. Because the expense exceeds $200, she also needs manager approval.", "relevant_passage_id": "handbook_expenses_c1_s1"}
```

Each question deliberately avoids restating the section heading or distinctive vocabulary from its gold passage. Run them through the BM25 check in §4.2 before shipping.

### 10.3 Validate

```python
from pathlib import Path
validate(Path("corpus/test.jsonl"), Path("qa/test.jsonl"))
```

Expected output:

```
=== DISTRIBUTION ===
Corpus passages: 8
QA questions: 5
Passages used as gold: 5 (62.5%)
Passages never used: 3
Questions per passage:
  1 question(s): 5 passages

✓ Structural validation passed.
```

### 10.4 Test loading

```python
from datasets import load_dataset

corpus = load_dataset("your-org/your-dataset", name="corpus", split="test")
qa = load_dataset("your-org/your-dataset", name="qa", split="test")
print(f"Corpus: {len(corpus)} passages")
print(f"QA: {len(qa)} questions")
assert set(qa["relevant_passage_id"]).issubset(set(corpus["id"]))
```

---

## 11. Adversarial review checklist

The questions a hostile reviewer asks. Get ahead of them.

**Provenance**
- [ ] Source documents identified and their licence stated.
- [ ] Source licence is compatible with your release licence.
- [ ] Required upstream citations included (Legal RAG Bench paper, MLEB).

**Corpus**
- [ ] Hierarchy-encoding string `id` on every passage.
- [ ] Token cap and tokenizer pinned and documented.
- [ ] No stubs, no boilerplate, no duplicates.
- [ ] Footnote references all resolve.

**QA**
- [ ] String `id` and string `relevant_passage_id` on every row.
- [ ] Every `relevant_passage_id` exists in the corpus.
- [ ] Coverage table across topics and question types.
- [ ] Inter-annotator agreement ≥90% on a 10% sample.
- [ ] BM25 lexical-dissimilarity check passes.
- [ ] Expert-knowledge bar defined and met.

**Answers**
- [ ] Every answer derivable from the gold passage alone.
- [ ] No yes/no, one-word, or multiple-choice answers.

**Methodology (if running evaluations)**
- [ ] Pipeline hyperparameters pinned.
- [ ] Judge model and prompts published verbatim.
- [ ] Full factorial of embedders × LLMs.
- [ ] All three binary outcomes recorded per (question, embedder, LLM).
- [ ] Raw retrieval outputs and model responses released.

---

## 12. Quick reference

| Aspect | Requirement | Example |
|--------|-------------|---------|
| **Corpus fields** | `id` (str), `title` (str), `text` (str), `footnotes` (str\|null) | `{"id": "1.1-c1-s1", "title": "1.1 ...", "text": "...", "footnotes": null}` |
| **QA fields** | `id` (str), `question` (str), `answer` (str), `relevant_passage_id` (str) | `{"id": "q-001", "question": "...", "answer": "...", "relevant_passage_id": "1.1-c1-s1"}` |
| **ID format** | Stable, semantic, hierarchy-encoding string | `policy_2024_sec3_2_c1_s1` |
| **Passage per Q** | Single string, not array | `"doc_1"`, not `["doc_1", "doc_2"]` |
| **Token cap** | ≤512 under pinned tokenizer (Kanon in original) | Pin `cl100k_base` if not using Kanon |
| **Footnotes when absent** | `null`, not `""` | `"footnotes": null` |
| **Answer style** | Self-contained, long-form, corpus terminology | "Employees get 20 days per year, accruing monthly." |
| **HuggingFace configs** | `corpus` and `qa` | `load_dataset(..., name="qa")` |
| **Split** | Only `test` | `split="test"` |
| **Question authoring** | Sample passages first, hand-write, lexically dissimilar, expert-knowledge | Verify with BM25 rank check |
| **Methodology** | Correctness × Groundedness × Retrieval, decomposed in that order | Pin LLM-as-judge model and prompts |

---

## References and citations

```bibtex
@misc{butler2026legalragbench,
  title={Legal RAG Bench: an end-to-end benchmark for legal RAG},
  author={Abdur-Rahman Butler and Umar Butler},
  year={2026},
  eprint={2603.01710},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2603.01710}
}

@misc{butler2025massivelegalembeddingbenchmark,
  title={The Massive Legal Embedding Benchmark (MLEB)},
  author={Umar Butler and Abdur-Rahman Butler and Adrian Lucas Malec},
  year={2025},
  eprint={2510.19365},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2510.19365}
}
```

Both are required if you cite Legal RAG Bench, per the Isaacus dataset card.

**Sources consulted in writing this guide:**

- Hugging Face dataset card: `https://huggingface.co/datasets/isaacus/legal-rag-bench`
- Isaacus blog post: `https://isaacus.com/blog/legal-rag-bench`
- arXiv 2603.01710 (Butler & Butler, 2 March 2026)
- Hugging Face data viewer for the `corpus` subset (to verify `title` and `footnotes` field presence beyond what the README documents)