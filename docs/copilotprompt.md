# Copilot Prompts: Generate Sample Data in Legal RAG Bench Format

Three prompts to paste into Microsoft 365 Copilot Chat. Use **Web mode** unless you're referencing a file in OneDrive/SharePoint.

Run each prompt in a **new chat** — don't reuse threads, context bleeds.

---

## Target format (what you're producing)

Two JSONL files. One JSON object per line. Straight quotes only.

**`corpus/test.jsonl`** — one passage per line:
```json
{"id": "hr_handbook_3_2_c1_s1", "title": "3.2 Leave Entitlements", "text": "## Annual Leave\n\n20 days per year...", "footnotes": null}
```

**`qa/test.jsonl`** — one question per line:
```json
{"id": "q-001", "question": "Chen wants to take a fortnight off in three weeks...", "answer": "Yes. For annual leave longer than three days, two weeks' notice is required...", "relevant_passage_id": "hr_handbook_3_2_c1_s1"}
```

All fields are strings. `footnotes` is `null` when absent (not `""`). `relevant_passage_id` is a single string, never a list.

---

## Prompt 1 — Chunk a source document into corpus rows

```
Split the source document below into passages for a RAG evaluation corpus.
Output JSONL — one JSON object per line.

Each passage must have these four fields:
- "id": string of the form "[doc_shortname]_[section]_c[chunk#]_s[subchunk#]",
  e.g. "hr_handbook_3_2_c1_s1". Increment c when starting a new chunk within a
  section; increment s within a chunk.
- "title": string, the nearest section heading
- "text": string, Markdown-formatted (preserve #, ##, lists, bold, italic)
- "footnotes": string with Markdown footnote definitions ([^1]: ...), or null
  if there are none. Use null, not empty string.

Chunking rules:
- Aim for 200–400 words per passage. Break long sections at paragraph or list
  boundaries.
- Drop these — do NOT include as passages:
  * Headings with no body content
  * "Welcome to..." pages, copyright notices, tables of contents
  * Page headers and footers
  * Anything shorter than two sentences

Output rules:
- JSONL only. No code fences. No commentary before or after. No "Here's your..."
- STRAIGHT double quotes (") only. No smart quotes (" ").
- If the document is long, output the first 10 passages, then stop and wait
  for me to type "continue".

SOURCE DOCUMENT:
[PASTE DOCUMENT TEXT OR REFERENCE A FILE WITH /]
```

---

## Prompt 2 — Draft a question for ONE passage

Run this once per passage. You pick which passages to use — don't ask Copilot to choose.

```
Draft ONE question and ONE answer about the passage below, in the Legal RAG
Bench format.

Hard rules:
1. The answer must be derivable from this passage ALONE. Do not add facts from
   your own knowledge, even if you think they're correct.
2. The question must be lexically dissimilar from the passage. Do NOT reuse
   distinctive nouns, verbs, or phrases from the passage. Do NOT reference the
   section number, heading, or position.
3. The question must be a real-world scenario or hypothetical, not a fact
   lookup. ("What does section 3.2 say about X" is forbidden.)
4. The question should require expert-level domain knowledge in [DOMAIN] to
   answer well — not just reading comprehension.
5. The answer must be 1–3 sentences of long-form prose. NOT yes/no, NOT one
   word, NOT a multiple-choice letter.
6. Use the passage's own terminology in the answer. If it says "associates,"
   do NOT switch to "employees."

Output rules:
- Output ONLY a single JSON object. No code fences. No preamble. No postscript.
- STRAIGHT double quotes only.
- Fields, in this order:
  * "id": string like "q-001" (I'll renumber later)
  * "question": string
  * "answer": string
  * "relevant_passage_id": string — copy the id from the passage below EXACTLY

After the JSON, write 2 sentences explaining how the question avoids restating
vocabulary from the passage.

PASSAGE:
{
  "id": "[PASTE PASSAGE ID]",
  "title": "[PASTE TITLE]",
  "text": "[PASTE TEXT]",
  "footnotes": [PASTE FOOTNOTES OR null]
}
```

---

## Prompt 3 — Stress-test a draft (run in a NEW chat)

Don't run this in the same thread as Prompt 2 — Copilot will defend its own work.

```
Review this question for a RAG evaluation dataset. Be ruthless. Your goal is
to find reasons to reject it, not to validate it. Answer each check with PASS
or FAIL and a one-sentence reason.

1. Does the question reuse any distinctive noun, verb, or phrase from the passage?
2. Does the question reference a section number, heading, or document position?
3. Is the answer fully derivable from the passage, with no invented facts?
4. Does the answer use the passage's terminology (not synonyms)?
5. Is the answer 1–3 sentences of long-form prose (not yes/no, not one word)?
6. Could a layperson answer this with the passage in front of them, or does it
   genuinely require domain expertise?
7. Is the question phrased as a real-world scenario, not a fact lookup?

PASSAGE:
[paste]

DRAFT QUESTION: [paste]
DRAFT ANSWER: [paste]

After the 7 checks, give an overall verdict: SHIP, REVISE, or REJECT.
```

If you get 2+ FAILs, throw the draft away and rerun Prompt 2 with a different angle.

---

## After Copilot finishes — clean before saving

Copy the output into a **plain-text editor** (VS Code, Notepad++, Notepad — *not Word*, which re-introduces smart quotes). Check for and fix:

- Smart quotes `"` `"` → straight `"` (find-and-replace)
- Code fences ` ```json ` … ` ``` ` → delete
- `**"id"**` → `"id"` (Markdown bold around field names)
- `"footnotes": ""` → `"footnotes": null`
- `"id": 1` → `"id": "q-001"` (Copilot drifts to integers — must be strings)
- `"relevant_passage_id": ["..."]` → `"relevant_passage_id": "..."` (never a list)
- Made-up `relevant_passage_id` values that don't exist in your corpus

Save one JSON object per line in `corpus/test.jsonl` or `qa/test.jsonl`.

---

## When to hand off

Every 10 questions, send both files to your Python-equipped teammate to run the validator. Don't wait until 100.