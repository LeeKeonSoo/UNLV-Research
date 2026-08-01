# Temporal-Code Stage-B Blind Review Guide

## Purpose

This optional diagnostic tests whether the Stage-B Quality and
soft-Redundancy proxies agree with independent human judgments on real
code-corpus chunks. It is not canonical selector ground truth, cannot tune or
promote Stage B, and cannot block corpus expansion or Stage-C entry.

Reviewers must not inspect:

- Stage-B scores or objective values
- selected or random-baseline membership
- repository identity, paths, or sampling strata
- the separate blind-review key
- Utility, benchmark, or training outcomes
- the other reviewer's labels before both reviews are frozen

The review is not a test of whether the code compiles or whether the reviewer
personally likes the project. Judge whether the chunk should be preserved in a
limited-budget language-model training corpus.

## Quality Labels

`preserve`

The chunk contains substantive, learnable technical information or behavior
that should usually survive limited-budget selection. Examples include a
meaningful implementation, a focused test with observable behavior, or
specific technical documentation.

`neutral`

The chunk is usable and not harmful, but its incremental training value is
unclear or modest. Examples include ordinary glue code, broad setup code, or
documentation that is correct but only weakly informative by itself.

`downrank`

The chunk is structurally valid but has little standalone learning value or is
dominated by boilerplate, pass-through behavior, trivial declarations,
verbose filler, or context-free fragments.

## Redundancy Labels

Judge redundancy relative to the other records in the assigned blind packet.
Identifier renaming or superficial wording changes do not make a chunk unique.

`unique`

No close substitute with substantially the same behavior, structure, or
technical information is apparent in the packet.

`related`

The chunk belongs to a repeated family or overlaps another record, but retains
meaningful distinct behavior or information.

`saturated`

One or more close substitutes provide substantially the same behavior,
structure, or information; retaining every instance would waste a limited
training budget.

## Confidence Labels

- `high`: the label follows clearly from the visible chunk and packet context
- `medium`: the label is likely but depends on missing surrounding context
- `low`: the chunk cannot be judged reliably from the visible evidence

Low confidence is not a reason to force `neutral`; choose the best Quality and
Redundancy labels and separately record low confidence.

## Workflow

Check progress:

```powershell
conda run --no-capture-output -n research python 84_manage_temporal_code_stage_b_review.py status --packet reviewer_a
```

Show the next incomplete record:

```powershell
conda run --no-capture-output -n research python 84_manage_temporal_code_stage_b_review.py show --packet reviewer_a
```

Record a label:

```powershell
conda run --no-capture-output -n research python 84_manage_temporal_code_stage_b_review.py label --packet reviewer_a --review-id REVIEW_ID --quality preserve --redundancy unique --confidence high
```

Freeze a completed independent review:

```powershell
conda run --no-capture-output -n research python 84_manage_temporal_code_stage_b_review.py freeze-review --packet reviewer_a --reviewer-attestation REVIEWER_ID --attest-independent --attest-no-key
```

After both reviews are frozen, activate disagreement-only adjudication:

```powershell
conda run --no-capture-output -n research python 84_manage_temporal_code_stage_b_review.py activate-adjudication
```

The adjudicator follows the same label definitions, sees only disagreement
records, and must not inspect the hidden key.
