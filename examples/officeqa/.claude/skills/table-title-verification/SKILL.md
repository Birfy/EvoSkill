---
name: table-title-verification
description: Invoke when a source file contains multiple named or numbered data tables and the question describes a specific report, trust fund, account, program, or entity type whose name can be matched against formal table titles in the file's table-of-contents or heading hierarchy.
---

## When To Use

The source file has a table-of-contents, index, or heading hierarchy listing multiple named or numbered data tables (e.g., "GA-III-1 ... GA-III-9", "FCP-VI-1 / FCP-VI-2 / FCP-VI-3"), AND the question describes the target data source in terms of a specific report, trust fund, account, program, or entity type whose name semantically corresponds to one or more of those formal table titles. Also activate when the task prompt says "prefer the table whose title and period match the question over nearby summary tables."

Common activation patterns:

- A Treasury Bulletin lists 9 GA-III tables for different trust funds; the question asks about "Federal Hospital Insurance Trust Fund financial operations."
- A document contains FCP-VI-1 (weekly), FCP-VI-2 (monthly), and FCP-VI-3 (quarterly); the question specifies a weekly period.
- A source has adjacent tables for "Customs Duties and Taxes — Values" vs. "Customs Duties and Taxes — Tariff Schedules"; the question asks for value data.

## Do Not Use

- The source file contains only one data table with no other table that could plausibly be confused for it.
- The question explicitly specifies a table name or number with no ambiguity (e.g., "in Table GA-III-4, what is the balance?"). In this case the agent already has the correct table.
- The source file lacks any table-of-contents, heading hierarchy, or formally named/numbered tables — tables are identified only by surrounding prose.
- The task is a purely textual lookup with no tabular data involved.
- The question does not reference any specific entity, program, fund, or report name that could be matched against table titles (e.g., "What is the value in row 5?" with no entity context).

## Failure Mechanism

When a source file contains multiple tables with similar column structures but different entity coverage, an agent that jumps directly to reading values without first identifying which table matches the question's entity risks extracting from the wrong table. For example, a Treasury Bulletin document may contain nine GA-III tables covering distinct trust funds (Old-Age and Survivors Insurance, Disability Insurance, Hospital Insurance, Supplementary Medical Insurance, etc.). Each table has the same column layout (receipts, outlays, balance, etc.) but covers a different fund. Extracting values from GA-III-3 instead of GA-III-4 produces a correct-looking but entity-wrong answer. This skill prevents that by gating extraction behind a required table-title matching step.

## Procedure

1. **Scan the table-of-contents, index, or heading hierarchy** of the source file. Identify every named or numbered data table and record each table's formal title exactly as it appears. These are your candidate tables.

2. **Extract the entity description from the question.** Identify the specific report, trust fund, account, program, or entity type the question asks about. This is the target entity — the subject that the requested data must describe.

3. **Match the target entity against candidate table titles.** Compare semantically, not by exact string equality:
   - An exact substring match is ideal but not required.
   - A partial overlap is sufficient (e.g., question says "Federal Hospital Insurance" and table title is "GA-III-4 — Federal Hospital Insurance Trust Fund").
   - When multiple tables share overlapping keywords, select the table whose title contains the most specific match to the target entity. Prefer "Federal Hospital Insurance Trust Fund" over "Federal Old-Age and Survivors Insurance Trust Fund" when the question asks about hospital insurance.
   - Weight matches on the entity name higher than matches on generic terms like "Table," "Statement," "Summary," or "Report."
   - When the question specifies a temporal granularity (weekly, monthly, quarterly, annual) and multiple tables differ only by period, match on the period as well.

4. **Name the selected table explicitly in your reasoning trace** before extracting any values. Use the table's formal title and identifier as it appears in the source. Example: "I will use Table GA-III-4 — Federal Hospital Insurance Trust Fund." This step is mandatory — do not proceed to extraction without it.

5. **Proceed with value extraction** from the named table. Subsequent row/column matching, entity selection within the table, and computation steps proceed normally. This skill only governs which table to use, not how to use it.

If no table title matches the target entity, describe the ambiguity in the reasoning trace and proceed with the closest available match, noting the uncertainty.

## Invariants

- Must NOT block or gate-keep extraction when only one data table exists in the source file. Fast-path skip.
- Must accept partial and fuzzy title matches. Do not require exact string equality between the question's phrasing and the formal table title.
- Must NOT interfere with subsequent row/column matching (entity-qualifier-match) or computation steps (computation-trace-verification) within the correctly selected table.
- Must allow selecting and using multiple tables when the question requires data from more than one table. Name each selected table before extracting from it.
- Must proceed with the best available match even when no table title is a perfect semantic fit. Document the match rationale and continue rather than halting.

## High-Risk Operations

- Rejecting all candidate tables when a plausible but imperfect title match exists — this would block completion entirely.
- Insisting on an exact title match — table formal names frequently use longer, more bureaucratic phrasing than questions (e.g., question says "hospital insurance fund operations," table says "GA-III-4 — Federal Hospital Insurance Trust Fund").
- Picking the first table in source order rather than the best semantic match — table ordering in a document is not a signal of relevance.

## Regression Risks

- Could cause the agent to reject a correct table if the title-match threshold is applied too strictly. A question asking about "Federal Hospital Insurance" must be accepted as matching a table titled "GA-III-4 — Federal Hospital Insurance Trust Fund."
- Could add unnecessary overhead on single-table documents where table identification is trivial. The procedure must fast-path skip when only one table exists.
- Could cause confusion with documents where tables lack formal titles or numbers and must be identified by surrounding prose context. The skill should not activate in that scenario.
- Could interfere with multi-table queries by forcing selection of exactly one table. The skill must allow naming and using multiple tables when the question requires it.
