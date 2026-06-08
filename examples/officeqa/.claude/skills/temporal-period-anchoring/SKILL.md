---
name: temporal-period-anchoring
description: Invoke when computing statistics across ≥2 distinct time periods where source-to-period mappings are ambiguous or span multiple files.
---

## When To Use

Activate this skill when ALL of the following conditions are true:

1. The question requests data from **N≥2 distinct time periods** (calendar months, weeks, quarters, fiscal periods, or named date windows) that must be extracted from source files.
2. At least one of these ambiguity factors is present:
   - **Publication-date / data-date mismatch**: the source file's label (e.g., "November 1980 bulletin") does not directly name the target data period (e.g., "October 1980" or "prior-year November 1979").
   - **Windowing rules**: periods are defined relative to an event or boundary (e.g., "the 5 weeks preceding event X", "the 3 months following Q2", "Feb through May").
   - **Non-trivial exclusion/inclusion**: some periods within a range are explicitly excluded (e.g., "all months except the event month", "every other week starting from...").
   - **Multiple source files**: periods span more than one source document, each covering different date ranges.
3. The requested output involves **arithmetic across periods** (sum, average, difference, ratio, growth rate, comparison) — not just listing individual period values independently.

## Do Not Use

Skip this skill when:

- The question requests a **single annual, fiscal-year, or single-period value** from one clearly-labeled row (no period disambiguation needed).
- All requested periods are **consecutive rows in a single table** with identical time granularity (e.g., "January through June" in a monthly table where each month is one row), with **no exclusion rules** and **no publication-date/data-date gap**.
- The entire source table covers **exactly the requested period** with no sub-period selection needed (e.g., "extract the 1940 annual total" from a table whose sole row is "1940").
- The task involves only **within-source column selection** (choosing between columns inside one already-identified row). That is governed by semantic-entity-binding.
- The agent is comparing table structures across publication vintages. That is governed by cross-vintage-structural-verification.

## Failure Mechanism

If the agent attempts to compute any arithmetic result (sum, average, difference, ratio, growth rate) across periods without first enumerating every period-to-source mapping, the skill MUST:

1. **Halt computation** — do NOT produce a numeric result until the enumeration is complete and verified.
2. **Require the agent to produce an explicit period map** with these columns for every target period:
   - Target period (e.g., "November 1979", "Week of Feb 12, 2020")
   - Source file (e.g., `treasury_bulletin_1980_11.txt`)
   - Table or section name within that file
   - Row identifier(s) (line number, row label, or index) for that period's data
3. **Verify period count** against the question: does the map contain exactly the N periods requested? Are any missing or duplicated?
4. **Verify boundary rules**: if the question specifies "preceding 5 weeks" or "months Feb-May", confirm that the first and last period in the map match those boundaries and that any exclusion rules (e.g., "excluding the event week") are applied.
5. Only after the period map passes all checks, allow arithmetic to proceed.

The failure mode is **off-by-one window error, period omission, or publication-date/data-date confusion**: the computed result may be arithmetically correct on the wrong set of rows, producing a plausible-looking but wrong answer.

## Procedure

When the agent is about to compute any temporally-scoped statistic across N≥2 periods:

### Step 1: Parse Temporal Requirements

Extract from the question:
- The exact list of target periods (with calendar anchors where specified)
- Any event date that defines period boundaries
- Any inclusion/exclusion rules ("preceding", "following", "during but not including", "every other")
- Whether the period labels match source file labels or require translation (e.g., "November 1979 calendar" → "November 1980 bulletin" because prior-year data appears in the following year's publication)

### Step 2: Produce the Period Map

For every target period, enumerate:

| # | Target Period | Source File | Table/Section | Row Identifier |
|---|---|---|---|---|

The Row Identifier must be specific enough to locate the value: a line number, a labeled row key, or a structured index. Do not use vague references like "the third row from the bottom."

### Step 3: Validate the Map

Run these checks **before** reading actual data values:

1. **Count check**: Does the map have exactly the number of periods the question requires? Count rows in the map.
2. **Boundary check**: Do the first and last periods match the question's start and end boundaries? If "preceding 5 weeks before event E", is the earliest week exactly E−5 weeks and the latest exactly E−1 week?
3. **Exclusion check**: If the question says "excluding period X", is X absent from the map? If "including period Y", is Y present?
4. **Gap/overlap check**: Are periods contiguous where required? No unintended gaps or double-counting?
5. **Publication-date alignment check**: For each source file, confirm whether its publication date/label differs from the data periods it contains. If a bulletin labeled "November 1980" contains October 1980 data (as is common), document this explicitly to prevent using the wrong row.

### Step 4: Gate Arithmetic

Only after all checks in Step 3 pass, proceed to:
- Reading the identified rows
- Extracting the specific values
- Performing the requested arithmetic (sum, average, difference, ratio, etc.)

### Step 5: Document the Mapping

In the final answer, include the period map (or a summary) so the mapping is auditable. This is not optional — it provides traceability from each computed value back to its source location.

## Invariants

- Must NOT alter which tables, rows, or columns are selected within a source (preserves semantic-entity-binding).
- Must NOT alter arithmetic procedures or validation of computed results (preserves computation-verification).
- Must NOT alter cross-source value reconciliation or conflict resolution logic.
- Must NOT alter cross-vintage structural comparison when multiple publication vintages are combined (preserves cross-vintage-structural-verification).
- Must NOT block computation when the period mapping is trivially obvious and already correctly performed by the agent. The skill gates only when mapping ambiguity exists — if the agent already produces an explicit period map voluntarily, the skill is satisfied.
- The period map is a prerequisite to computation, not a replacement for any existing verification step.

## High-Risk Operations

These operations are forbidden without first completing the period map and validation:

- **Computing a sum, average, or aggregate** across periods whose source mapping has not been enumerated.
- **Computing a ratio or growth rate** between two periods without confirming which source rows correspond to the numerator and denominator periods.
- **Comparing values** from different periods when the source file publication date could be mistaken for the data date (e.g., using the "November 1980" bulletin header value for November 1980 when the bulletin actually reports October 1980 data).
- **Windowing around an event** without confirming the event's exact date anchors to a specific source row and that the pre- and post-event windows are correctly bounded.
- **Summing "N months of data"** by counting rows when some months may be missing, split across files, or reported with a publication lag.

## Regression Risks

| Risk | Mitigation |
|---|---|
| Over-triggering on simple two-period tasks where the mapping is trivially obvious (e.g., two consecutive months in the same table, same file, no exclusion rules, no publication lag) | The "Do Not Use" clause excludes consecutive-row, same-file, identical-granularity cases. If the agent already correctly performs the mapping implicitly without ambiguity, the skill does not intervene — it only gates when ambiguity is present or the agent skips mapping. |
| Adding unnecessary enumeration overhead for single-period queries where only the publication date differs from the data date | The N≥2 threshold prevents activation on single-period tasks. If the agent incorrectly maps a single period, that is a semantic-entity-binding error, not a temporal-anchoring error. |
| Disrupting agents that already perform correct temporal mapping as part of their standard workflow | The skill's gate is satisfied by ANY explicit period-to-source mapping, whether produced because of the skill or produced voluntarily. If the agent already enumerates periods before computing, no additional step is required. |
| Conflicting with cross-vintage-structural-verification when multiple bulletins are used | This skill enumerates which periods come from which files but does NOT check whether table structures match across those files. If the agent later combines values into a time series from different vintages of the same table, cross-vintage-structural-verification handles the structural equivalence check independently. |
