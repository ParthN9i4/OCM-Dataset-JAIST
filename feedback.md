# Working-Style Feedback (extracted from real corrections)

This file captures corrections made to Claude's work during the OCM catalyst-transfer project, and
the concrete habits that should follow from them. Load this at the start of any session (this file is
referenced from `CLAUDE.md`) — it isn't project-specific advice, it's about *how* to work.

## Corrections made, chronologically, with the underlying lesson

1. **Email draft was too ML-jargon-heavy** for a non-expert audience returning after 6 months.
   → Rewrite in plain language whenever the stated audience isn't ML-literate; don't default to
   technical register just because the underlying work is technical.

2. **Same email, framing made steady work sound like "didn't work for a while."**
   → Reframe ruled-out approaches as deliberate diagnostic work that earned an insight, not as time
   wasted. Ruling something out is progress; say so plainly.

3. **Same email again — novelty was buried**, risking "a lot of time for little work."
   → Give the actual novel contribution its own explicit, unmissable paragraph, and compress the
   "here's what didn't work" section further than feels natural. (This took **3 separate corrections**
   on one email — tone and emphasis risk should be self-checked *before* presenting a draft, not only
   after the user has to flag it three times.)

4. **A new-repo README was flagged as "not updated at all"** even though nothing had technically
   changed (notebook was byte-identical, no new experiments existed).
   → The real issue was terminology drift: an independently-written paraphrase used different names
   ("Naive merge" instead of "Direct merge"), never used the project's actual coined term ("PFT"),
   didn't group related methods the way the authoritative document did ("Selective merge" for
   DRST+KMM), and had an unsourced novelty paragraph instead of the citation-backed original.
   **When summarizing/deriving from a source that's already been authored and externally reviewed,
   pull terminology and structure directly from that source — don't re-paraphrase from memory.**

5. **"Is it 1.907 or 1.916?"** surfaced two distinct problems:
   - Multiple legitimately-different numbers (a single-seed run vs. a 10-seed average) existed
     side by side without being clearly labeled as different things — ambiguity that read as an error.
   - An earlier factual claim (numbers found in a PDF) was a **false positive** from running `strings`
     on a compressed/binary format, which I had to retract after using a more reliable extraction.
   → Label what differs whenever near-identical numbers coexist, *before* someone has to ask. Never
   trust raw byte-scanning on structured binary formats (PDF/docx/xlsx) for real content — use proper
   structured extraction (e.g. zipfile+XML for Office formats), and say so explicitly if forced to fall
   back to something weaker.

6. **User supplied the actual document that was sent externally** and asked "are you sure this is
   correct?" — it turned out my own repo copy had silently forked from what was really sent (different
   RMSE values, different described configuration). The sent document itself also had an internal
   arithmetic slip (a stated percentage that only recomputes correctly from a different number than the
   one printed beside it), which I hadn't caught proactively.
   → **Treat "which artifact is the actual ground truth" as a first-class question** whenever an
   internal working copy and an externally-sent version of ostensibly the same document both exist —
   seek out the sent one rather than trusting the local copy by default. Recompute derived quantities
   (percentages, deltas) from stated base numbers as a standing sanity check before they go into any
   new document — cheap, and it catches copy/paste and stale-edit slips.

7. **"Keep the readme short... no need to keep credits or future work in detail."**
   → The first worknote-aligned pass mirrored the source document's full depth. Match the depth to
   what the *target artifact type* actually needs (a scannable README is not a full report), not to
   the depth of the source being summarized.

## Verification & rigor lessons

- **What worked and should be the default:** for code/notebook work, verification meant actually
  executing end-to-end, diffing file mtimes, and re-extracting real stored outputs before declaring
  something done — never just asserting it. Hold prose/summary documents to the *same* standard, not a
  lighter one.
- **What didn't work:** reporting a factual claim from an unreliable extraction method without flagging
  its limits. If a check is weak, say so in the same breath as the finding — don't present a shaky
  result with the same confidence as a solid one.
- **What worked well:** when a user says something is wrong and my own checks show no technical change,
  asking what specifically looks wrong (rather than re-asserting confidence) resolved the situation
  fast and avoided guessing at the wrong fix.
- **From earlier in this project:** a "prior" model trained on data that overlapped test rows produced
  a leakage-inflated headline number that propagated across many downstream artifacts before an
  independent re-verification script caught it. Numbers hand-copied into multiple files (rather than
  derived from one script/notebook) drifted from the real experiment over time and had to be
  reconciled after the fact.
- **From earlier in this project:** creating a new file to replace an old one without visibly flagging
  the replacement caused the user to think nothing had changed, since they kept looking at the old one.

## Do differently next time — the actionable list

1. Derive terminology/structure from the authoritative, already-reviewed source document — don't
   independently re-paraphrase from memory.
2. Label differences between near-identical numbers (different run, different config) proactively,
   before they can look like contradictions.
3. Recompute derived percentages/deltas from stated base numbers as a standing sanity check before
   reusing them in a new document.
4. Never trust `strings`/raw byte-scanning on structured binary formats for real content — use proper
   structured extraction, and explicitly flag it when falling back to something weaker.
5. When multiple copies of "the same" document might exist (internal working copy vs. what was
   actually sent), find and defer to the one that was actually sent/communicated.
6. Default to brevity for durable external-facing summaries (READMEs, etc.) unless told otherwise —
   match depth to the artifact type, not to the source it summarizes.
7. Tone-check external-facing prose (especially emails) for "does this make our effort sound
   wasted/small" before presenting it, not only after being told.
8. Keep one script/notebook as the single source of truth for any number reused across multiple
   artifacts; explicitly flag it when a file supersedes/replaces another rather than adding it quietly.
