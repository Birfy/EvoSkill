---
name: source-revision-authority
description: Invoke when multiple provided source files contain data for overlapping entity×time_period×metric combinations and the agent is about to finalize an extracted numeric value without having cross-checked all other files for revised or restated figures for the same data point.
---

## When To Use

Multiple source files are listed for a time-series or quantitative data extraction task, AND the agent extracts a value for a specific entity×time_period×metric from one file, AND at least one other listed source file also covers a time range that includes the same time period for the same metric. Activate before the agent finalizes or reports the extracted value — this is a pre-commit gate that requires cross-file comparison when source files overlap on the data point being extracted.

Common activation patterns:

- A task provides both an April 1948 Treasury Bulletin and a December 1952 Treasury Bulletin; the agent extracts FY1947 Army expenditures from the 1948 file. The 1952 file also covers FY1947 and may contain a revised figure.
- A task provides multiple annual reports from different years that both restate historical data for the same metric; the agent reads from an earlier report without checking the later report's restated values.
- A task provides preliminary and final releases of the same statistical table; the agent uses the preliminary release's figure without checking the final release for the same row and column.

## Do Not Use

- Only a single source file is provided for the metric in question — there is no other file to cross-check against.
- The question explicitly specifies which source file, publication date, or data vintage to use for each data point (e.g., "using the April 1948 Treasury Bulletin, what were Army expenditures in FY1947?").
- The values are identical across all files containing that entity×time_period×metric — the cross-check is a no-op.
- The overlapping files differ in reporting structure for the target metric (e.g., one file includes Air Force expenditures within Army while another separates them). Structural differences mean the conflict is a scope question, not a revision question — this is handled by column-scope-verification.
- All listed source files have non-overlapping time ranges for the target metric — there is no shared entity×time_period×metric to cross-check.
- The task does not involve numeric value extraction from tabular data (e.g., purely textual lookup, document classification, summarization).
- The agent has already performed the cross-file comparison and documented the result in the reasoning trace before finalizing the value.

## Failure Mechanism

When multiple source files are provided that overlap on time periods for the same metric, an agent may extract a value from the earliest or most readily accessible file and treat it as final without checking whether a later file contains a revised or restated figure for the same entity×time_period×metric. Government statistical publications routinely restate historical data in later editions as more complete accounting becomes available. A Treasury Bulletin from 1948 reports preliminary FY1947 figures; a 1952 Treasury Bulletin restates those same FY1947 figures with 4+ additional years of reconciled data. The agent that uses the older preliminary figure produces a value that conflicts with the authoritative restated figure in the later publication, yielding a numerically wrong answer that no amount of correct computation within the wrong source can fix.

This skill prevents that by gating value finalization behind a required cross-file revision check. Before the agent reports any extracted numeric value, it must verify that no other provided source file contains a conflicting value for the same entity×time_period×metric.

## Procedure

1. **Identify the data point being extracted.** Before extracting a value, write down the entity (e.g., "Army expenditures"), the time period (e.g., "FY1947"), and the metric (e.g., "net expenditures in millions of dollars"). This is the target data point.

2. **Check every other listed source file for the same data point.** For each other source file provided in the task:
   - Does the file contain tabular data for the same metric at any time?
   - If yes, does the file's time range include the target time period?
   - If yes, locate the value for the same entity×time_period×metric in that file.

3. **Compare the values extracted from each file.** If only one file contains the target data point, no conflict exists — proceed normally with that value. If multiple files contain the target data point and all values agree, the check is a no-op — proceed with any source, noting the agreement in the reasoning trace.

4. **When values conflict, check reporting structure before resolving.** For each file with a conflicting value, determine whether the reporting structure is consistent across files:
   - Do the files use the same entity decomposition? (e.g., both include Air Force within Army for FY1947, or both separate them)
   - Do the files use the same accounting basis? (e.g., both report net expenditures, both use the same fiscal year definition)
   - If the files have relevant footnotes or explanatory text, compare them to confirm the same entity×metric definition is in use.
   - A structural difference is a scope conflict, not a revision conflict — if the files define the entity differently, the resolution rule in step 5 does not apply. Instead, prefer the file whose entity definition matches the question's scope.

5. **Apply the revision-authority rule.** When the reporting structure is consistent across files (the entities are defined the same way) and values differ, prefer the value from the **later publication date**. Later publications incorporate revised, restated, or more completely reconciled data. Document this in the reasoning trace: "File A (published [earlier date]) reports [value A] for [entity×time_period×metric]. File B (published [later date]) restates this as [value B]. Both files define the entity consistently [cite evidence: matching footnotes, table structure, or explanatory text]. Using File B's revised figure of [value B] as the later and more authoritative source."

6. **Handle mixed-vintage extractions transparently.** When different data points in the same answer come from files with different publication dates (e.g., FY1940 from a 1948 bulletin, FY1947 from a 1952 bulletin), this is allowed. Note the mixed vintage in the reasoning trace so the comparison basis is documented.

7. **Proceed with computation using the resolved value.** After the cross-file check is complete and the authoritative value is selected, continue with any downstream computation steps. The skill's gate is satisfied once the cross-file comparison is documented in the reasoning trace.

## Invariants

- Must NOT force all data points into a single source file. Multi-source extraction is expected and allowed — extract FY1940 from file A and FY1947 from file B when each is the best (or only) source for that period.
- Must gate only the conflict-resolution step when values differ across files for the same data point, not the extraction itself.
- Must NOT trigger when values agree across overlapping files. The cross-check is a no-op — do not force the agent to prefer one file over another.
- Must NOT override an explicit source instruction in the question. If the question says "using the 1948 bulletin, ..." then use the 1948 bulletin.
- Must NOT apply the revision-authority rule when the reporting structure differs between files. A later file that splits an entity differently is not an "authoritative revision" — it is a different metric.
- Must allow mixed-vintage extractions when files cover non-overlapping time periods, as long as the mixed vintage is noted in the reasoning trace.

## High-Risk Operations

- Applying the revision-authority rule across files with different reporting structures. A 1952 file that separates Air Force from Army while a 1948 file combines them is not revising the figure — it is reporting a different entity. The rule must only apply when the entity definition is confirmed consistent.
- Overriding an explicit source specification in the question. If the question says "using the April 1948 bulletin," the agent must use that bulletin regardless of what later files say.
- Forcing all values to come from a single file when the best data spans multiple files. The skill must not penalize multi-source extraction.
- Accepting a later file's value without checking for structural consistency. Step 4 is mandatory before step 5.

## Regression Risks

- Could cause the agent to prefer a later file that uses a different reporting structure (e.g., a 1952 file that separates Air Force from Army when the question requires them combined). Mitigation: step 4 requires structural consistency check before applying the revision-authority rule.
- Could add overhead on tasks where the revision is immaterial to the answer. Mitigation: only activate when overlapping values actually differ — if they agree, the gate is a no-op.
- Could cause the agent to mix vintages inconsistently, producing an internally inconsistent comparison. Mitigation: step 6 requires documenting mixed-vintage extractions in the reasoning trace.
- Could cause the agent to re-read every file from scratch unnecessarily. The procedure should be applied when the agent is about to finalize a value, and only files known to cover the target time period need to be checked.
- Could conflict with table-title-verification or column-scope-verification if the cross-file check is applied before the correct table or column is identified within each file. The cross-file comparison should happen after within-file table and column selection are resolved but before the value is finalized.
