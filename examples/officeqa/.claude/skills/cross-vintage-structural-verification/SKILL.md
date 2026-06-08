---
name: cross-vintage-structural-verification
description: Invoke when combining data from the same named table across ≥2 bulletin vintages to produce a combined statistic or time series.
---

## When To Use

Activate this skill when ALL of the following conditions are true:

1. The agent is extracting values from the **same named table** (e.g., IFS-1, Table 3, Appendix B) across **two or more publication vintages** (different bulletin years, different source files).
2. The purpose is to **combine values** into a single time series, panel, or aggregate statistic (mean, sum, growth rate, standard deviation, etc.).
3. At least one of the following structural differences exists between the table versions, OR the agent has not yet verified that none exists:
   - Different **column count** in the data body
   - Different **header hierarchy** — sub-groupings, merged cells, indented sub-columns, parent-child header relationships
   - Different **footnote numbering** or **footnote count** — changed numbering schemes (e.g., "1/, 2/" vs "1, 2"), renumbered footnotes, or missing/reassigned footnote labels
   - **Revision markers** (r, p, e, ¹, ²) on any data row whose year appears in more than one vintage

The core insight: a column name match ("Total reserve assets") is NOT sufficient evidence of semantic equivalence when the enclosing table structure differs. The column definition may have changed between vintages.

## Do Not Use

Skip this skill when:

- All required values come from a **single source file** (no cross-vintage combination needed).
- The two table versions have **identical** column count, **identical** column header structure (same sub-groupings at every level), and **identical** footnote numbering (same count, same labels, same topical assignments) — indicating no methodological change.
- The task involves **only a single year's data** (no time series construction).
- The table is explicitly marked as a **continuation** (not a restatement or revision) of the same table series with identical layout.
- The agent is performing **within-source column selection** (e.g., choosing between "Total assets" and "Total liabilities" inside one bulletin). That is governed by semantic-entity-binding, not this skill.

## Failure Mechanism

If structural differences are detected between table versions during cross-vintage binding, the agent MUST:

1. **Halt extraction** — do NOT produce a combined numeric result (mean, sum, time series, etc.).
2. **Report the specific difference(s)** found, naming each divergent structural property with its values per vintage:
   - Column count: "Vintage A has N columns; Vintage B has M columns"
   - Header hierarchy: "Vintage A has sub-grouping X at level Y; Vintage B is flat"
   - Footnote numbering: "Vintage A uses footnotes a/-f/; Vintage B uses footnotes 1-4,6 (footnote 5 reassigned/removed)"
   - Revision markers: "Row for year Y in Vintage A is marked 'r' (revised); same row in Vintage B is final"
3. **Do NOT** rely on column-name equality alone to justify combining values.
4. **Recommend** using only the most recent vintage (prefer the later publication, as it typically reflects revised/final data) OR requesting user guidance on inter-vintage reconciliation.

The failure is silent-correctness: the combined statistic may appear numerically plausible (close to expected) but be derived from incompatible definitions, making it wrong in ways that are invisible without the structural check.

## Procedure

When the agent plans to extract and combine values from the same named table across ≥2 publication vintages:

### Step 1: Extract Structure Signatures

For each vintage of the table, record these four structural properties before reading any data values:

| Property | How to check |
|---|---|
| **Column count** | Count the number of data columns in the table body (not just the header row). Ignore row-label/description columns; count only value-bearing columns. |
| **Header hierarchy** | Map the full header structure. Identify multi-level headers, merged cells spanning multiple columns, indented sub-columns, and parent-child groupings (e.g., a "Gold stock" header with sub-columns "Volume" and "Value"). Record whether the structure is flat or hierarchical, and at what depth. |
| **Footnote scheme** | Count the footnotes. Record their numbering format (e.g., "1, 2, 3, 4, 5, 6" vs "1/, 2/, 3/, 4/, 5/, 6/"). Check whether any footnote labels present in one vintage are missing or reassigned in the other. |
| **Revision markers** | Check data rows for any revision flags (r, p, e, superscript numerals) on years that appear in both vintages. A revision marker on an overlapping year signals that the earlier vintage's value was preliminary. |

### Step 2: Compare Signatures

Compare the four properties pairwise between all vintages that will contribute values.

### Step 3: Decision

- **If all four properties match** across all vintages: the tables are structurally equivalent. Proceed with cross-vintage value binding. Note the structural equivalence confirmation in reasoning.
- **If any property differs**: halt. Report the differences (see Failure Mechanism). Do NOT produce a combined output. Either restrict to the most recent vintage or request user guidance.

## Invariants

- Must NOT prevent legitimate combining of data from identically structured tables across multiple bulletins (e.g., when a table's layout, headers, and footnotes remain unchanged between consecutive publications).
- Must NOT override or interfere with semantic-entity-binding's column-selection logic within a single source.
- Must NOT affect single-source computations or same-file extractions.
- Must apply **before** extraction for any cross-vintage data combination, not after values have already been committed to a combined result.
- Table-name matching (checking that both sources refer to the same named table) is a prerequisite for this skill but not part of its structural comparison.

## High-Risk Operations

These operations are forbidden without first completing the structural equivalence check:

- **Binding values** from the same named table across publication vintages into a single variable or time series.
- **Computing aggregate statistics** (mean, sum, standard deviation, minimum, maximum, growth rate) from cross-vintage data.
- **Treating column-name match as sufficient** evidence of semantic equivalence when the table structures differ.
- **Ignoring revision markers** on overlapping-year rows — a revised value in an earlier vintage may reflect a different methodology, not just a data correction.
- **Concatenating time series segments** where the breakpoint between vintages coincides with a structural change in the underlying table.

## Regression Risks

| Risk | Mitigation |
|---|---|
| Over-triggering on cosmetic differences (blank column added for spacing, OCR whitespace variation) | Only flag differences in *data-bearing* columns. Distinguish structural hierarchy changes from whitespace artifacts by examining whether the header text and sub-grouping relations are the same. |
| Slowing down cases where the agent correctly uses only the latest bulletin | The skill gates on cross-vintage combination. If the agent uses a single source, no structural check is triggered. |
| Flagging legitimate cross-vintage combinations where the table layout changed but the column definition remained the same (e.g., a column was split into sub-components) | When structural differences are flagged but the agent has external confirmation (e.g., a methodological note stating the definition is unchanged), the agent may proceed with explicit documentation of that confirmation. The check still fires — it just completes with a documented override rather than a halt. |
| Footnote renumbering from cosmetic reformatting (e.g., footnote symbols changed from numbers to letters) | If the footnote *count* and *topical assignments* (what each footnote describes) are identical, treat as cosmetic. Only flag when footnotes are added, removed, merged, or reassigned to different content. |
