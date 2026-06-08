---
name: column-scope-verification
description: Invoke when a question requests an aggregate or total measure scoped to an entity (e.g., "total balance of X Fund") and the source table contains columns whose headers include narrowing qualifiers naming sub-components, sub-accounts, sub-regions, or sub-programs of that entity.
---

## When To Use

The question asks for a quantitative attribute with an explicit entity scope qualifier (e.g., "total balance of the Unemployment Trust Fund," "net income from all operations," "revenue for the entire region") AND the source table has multiple data columns where some column headers contain entity names that are narrower in scope than the question's entity reference. The critical signal is a column header that names a sub-component, sub-account, sub-region, or sub-program of the entity the question asks about — not the entity itself. The agent is about to bind (or has already bound) the question's requested attribute to a specific column for value extraction.

Also activate when the agent reasons about whether a chosen column "equals" a derived value by checking only a small number of rows — the skill requires cross-row verification before accepting such equivalence.

Common activation patterns:

- Question asks for "total balance of the Unemployment Trust Fund" and a column header reads "Balance — Railroad Unemployment Insurance Account." "Railroad" is a narrowing qualifier — it names one sub-account, not the whole fund.
- Question asks for "total revenue from customs duties" and a column header reads "Customs Duties — Passenger Vehicles." "Passenger Vehicles" is a sub-category, not total customs duties.
- Question asks for "net income of the corporation" and a column header reads "Net Income — Consumer Division." "Consumer Division" is a sub-unit, not the whole corporation.

## Do Not Use

- The question explicitly names the sub-component by its exact column header or a close paraphrase (e.g., asks for "Railroad Unemployment Insurance Account balance" specifically). In this case the narrowing qualifier in the column header matches the question, so there is no scope mismatch.
- The table has a single unambiguous column for the requested metric — there is no choice to make.
- All columns in the relevant table section share the same entity scope and differ only by time period (e.g., columns are different fiscal years of the same measure, like "FY2020," "FY2021," "FY2022").
- The column header text is an exact semantic match for the question's requested attribute after accounting for minor wording variation (e.g., question asks for "total investments" and column says "Total investments").
- The agent has already verified the column selection by checking a derived relationship across multiple rows spanning the full data range (at least one pair where the values diverge), not just the first or last N rows.
- The task requires deriving the total from multiple component columns and no single column is being incorrectly treated as the total — the skill does not apply because there is no scope-mismatch risk to catch.

## Failure Mechanism

When a table has columns representing sub-components alongside columns representing aggregates or totals, an agent may incorrectly bind a question's broadly-scoped attribute (e.g., "total balance of the Unemployment Trust Fund") to a column whose header names a sub-component (e.g., "Railroad Unemployment Insurance Account"). The agent sees the sub-component column, notices it has plausible values, and treats it as the answer source without checking whether the column header contains narrowing qualifiers absent from the question.

This error is reinforced when the agent attempts to verify the binding by checking a derived identity (e.g., "does this column equal Total investments + Unexpended balance?") on only the first one or two rows. If the identity coincidentally holds on those rows (as Railroad Account = Total investments + Unexpended balance for FY1937-1938, diverging at FY1939), the agent generalizes incorrectly and extracts wrong values for all rows. The correct procedure is to derive the total from the aggregate sub-columns (here, Total investments + Unexpended balance) rather than borrowing a sub-component column that happens to equal the total on a subset of rows.

## Procedure

1. **Extract the question's requested attribute scope.** Identify what entity or aggregate the question asks about. Write it down explicitly. Example: "The question asks for the total balance of the Unemployment Trust Fund as a whole — the combined value across all accounts within the fund."

2. **Extract the scope of each candidate column header.** For each column that appears to contain the requested measure, parse the column header and identify any entity qualifiers. A qualifier is any word or phrase that narrows the column's scope to a sub-component, sub-account, sub-region, or sub-program. List each column with its scope annotation.

3. **Compare question scope to column scope.** For each candidate column:
   - If the column header contains an entity qualifier that names a sub-component narrower than the question's entity, flag it as a **scope mismatch**. The column measures a part, not the whole.
   - If the column header matches the question's entity without narrowing qualifiers, flag it as a **scope match**.
   - If the question asks for the total of an entity and no single column has matching scope, check whether the total can be derived by summing or combining multiple columns (e.g., "Total investments" + "Unexpended balance").

4. **If a scope mismatch is detected, do not silently proceed with the mismatched column.** Either:
   - Select a different column with matching scope, OR
   - Derive the total by combining the appropriate aggregate-level columns (not by borrowing a sub-component column), OR
   - Explicitly justify in the reasoning trace why the narrower column is being used despite the scope difference. This justification must explain what the column represents and why it is equivalent to the requested measure for all rows — not just a subset.

5. **Verify any derived equivalence across the full data range.** If you claim that a column "equals" a computed value (e.g., "Column X = Column Y + Column Z"), test that identity on multiple rows spanning at least three distinct periods that cover different conditions. Do not accept the equivalence after checking only the first one or two rows. If divergence is detected on any row, the equivalence is false and the column selection must be reconsidered.

6. **State the chosen column and the rationale explicitly** before extracting any values. Example: "I am extracting the total balance of the Unemployment Trust Fund as Total investments + Unexpended balance. I am NOT using the Railroad Unemployment Insurance Account column because it is a sub-component, not the fund total."

## Invariants

- Must not block valid extractions where the column header differs in wording but has identical semantic scope (e.g., "Balance Total" = "total balance," "Aggregate Revenue" = "total revenue"). The comparison is semantic, not lexical.
- Must not require perfect string matching between question terms and header text. Accept paraphrases, abbreviations, and minor wording variations that preserve semantic scope.
- Must allow multi-column derivations when no single column directly matches the question scope but the total can be derived from constituent aggregate columns. The skill does not require a single "total" column — it only requires that the agent not use a sub-component column as a shortcut for the total.
- Must not prevent the agent from reading and using the table when a scope mismatch is detected. The skill forces explicit justification and verification, not abandonment.
- Must not override the agent's judgment when the question itself specifies the narrowing qualifier. If the question asks for "Railroad Unemployment Insurance Account," then the "Railroad" qualifier in the column header is a feature, not a bug.

## High-Risk Operations

- Rejecting a column that IS correct simply because the header wording differs from question wording. Semantic equivalence must be the standard, not string identity.
- Treating a derived identity (Column = Sum of components) as proven after checking only the first few rows. Verification must span the data range.
- Forcing rejection of every column when none has a perfect scope match. When no single "total" column exists, the agent must derive the total from the data, not halt.
- Treating differently-worded column headers with the same scope as narrowing qualifiers (e.g., "Balance at End of Period" is not a narrowing of "total balance").

## Regression Risks

- Over-triggering on tables where column headers use differently-worded but semantically equivalent terms (e.g., "cumulative total" vs "total balance"). The skill must treat semantic scope, not wording, as the comparison criterion.
- False-positive scope mismatch when a header uses abbreviated terminology that omits part of the entity name but has the same scope (e.g., "Trust Fund Balance" when the entity is "Unemployment Trust Fund" — the qualifier is omitted, not added).
- Adding procedural overhead on tables with very few columns where scope is unambiguous. When the candidate column set is small and all columns are clearly distinguished by time or measure type, the scope check should be quick.
- Could cause the agent to reject a correct single column and instead add up sub-components incorrectly if the sub-components do not exhaust the total (e.g., there is an unlisted residual). Preferring an explicit "total" column over summing components is generally safe, but the skill should not force component summation when a reliable total column exists under a different name.
