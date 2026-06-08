---
name: cross-vintage-overlap-reconciliation
description: Invoke when extracting values from ≥2 source files with different publication dates for the same entity+period.
---

## When To Use

Activate this skill when ALL of the following conditions are true:

1. The question provides **≥2 source files** whose **publication dates differ** (different bulletin years, different release vintages).
2. The **target entity+time_period** (e.g., "Army expenditures, FY1947", "Gold reserves, December 1979") plausibly appears in more than one of the provided source files, OR the agent has not yet verified that it appears in only one.
3. The agent intends to **extract and bind** a numeric value for that entity+time_period from the source files.

The core failure: the agent opens only the first-provided source, finds the entity+period there, and binds the value without ever checking whether a later vintage contains a **restated/revised value** for the same entity+period. Later-vintage restatements supersede earlier preliminary values. This gate fires at **input-binding time** — before any value is assigned to a variable, used in arithmetic, or included in an answer.

## Do Not Use

Skip this skill when:

- Only **one source file** is provided, or all required data demonstrably comes from a single source.
- The entity+time_period is **provably unique to one file** after the agent has inspected ALL provided sources (a null scan result satisfies the gate — the gate only requires the scan, not that overlap exists).
- The question **explicitly names which source/vintage** to use for each data point (that instruction overrides the gate; follow the question's specification, not the "prefer later vintage" default).
- The task is purely **structural comparison of table layouts** across vintages (governed by cross-vintage-structural-verification).
- The task is **temporal period-to-source mapping ambiguity** — determining which source file contains data for which calendar period (governed by temporal-period-anchoring).
- The agent has **already inspected all provided sources** and documented that no overlapping entity+period values exist (the gate is satisfied once and does not re-trigger).

## Failure Mechanism

If the agent attempts to bind an entity+period value from a single source file without having inspected ALL provided source files for overlapping values, the skill MUST:

1. **Halt value binding** — do NOT assign the value to any variable, use it in arithmetic, or include it in a final answer.
2. **Require the agent to scan every provided source file** for the specific entity+period pair (e.g., search for "Army" near "1947" in each file). The scan must cover at minimum: the file's table of contents (if any), the data tables that contain the target entity, and any restatement or revision notes.
3. **Produce an overlap inventory** listing, for each entity+period pair the question requires:
   - Entity (e.g., "Army expenditures")
   - Period (e.g., "FY1947")
   - Source files checked (all provided files)
   - Value found in each file, or "Not present"
   - Publication date of each file that contains the entity+period
4. **Apply the vintage precedence rule**: when the same entity+period appears in multiple vintages, the **later publication's value takes precedence** because restated/revised values supersede earlier preliminary figures. The earlier vintage's value MUST NOT be used unless the question explicitly instructs otherwise.
5. **Document the vintage choice** — state which source was selected and why (later vintage, explicit question instruction, or sole source).
6. Only after the overlap inventory is complete and vintage choice documented, allow value binding to proceed.

The failure mode is **silent vintage staleness**: the bound value is numerically real (it was correctly read from a real source) but comes from a superseded preliminary vintage, producing a wrong answer that looks correct. This is undetectable without cross-source overlap checking.

## Procedure

When the agent has been provided ≥2 source files with different publication dates and intends to extract entity+period values:

### Step 1: Identify All Provided Sources

Before opening any source file, enumerate every source file the question provides, including:
- File path/name
- Publication date or vintage year (extract from filename, header, or metadata)

### Step 2: Enumerate Target Entity+Period Pairs

List every entity+time_period pair the question requires. Be specific:
- Entity: the named category (e.g., "Army expenditures", "National defense", "Gold stock")
- Period: the time scope (e.g., "FY1940", "calendar 1965", "December 1979")

### Step 3: Scan Every Source for Every Entity+Period

For each entity+period pair, scan every provided source file for that pair. Minimum scan steps:
- Search the file's table of contents or section headings for the relevant table
- Search within data tables for the entity name combined with the period label
- Check revision notes, footnotes, or methodological statements for restatement markers (e.g., "r" for revised, "p" for preliminary, superscript revision flags)

### Step 4: Produce the Overlap Inventory

Create a table with one row per entity+period pair:

| Entity | Period | File A (year) Value | File B (year) Value | ... | Selected Value | Rationale |
|---|---|---|---|---|---|---|

Rules for filling the table:
- If an entity+period appears in only one file, enter "Not present" for all other files. The sole source's value is the selected value.
- If an entity+period appears in multiple files, select the **later vintage's value** as the authoritative one. Document the earlier vintage's value in the table but do not use it.
- If the question explicitly instructs which vintage to use, that instruction overrides the "prefer later vintage" default.

### Step 5: Gate Value Binding

Only after the overlap inventory is complete for every entity+period pair, proceed to:
- Binding the selected values to variables
- Using those values in arithmetic or comparisons
- Including values in the final answer

### Step 6: Document in Final Answer

Include the overlap inventory (or a summary) in the reasoning or appendix of the final answer. State explicitly which vintage was used for each entity+period value and why.

## Invariants

- Must NOT prevent single-source extractions when only one source is provided or all data resides in one file.
- Must NOT override explicit question instructions about which vintage or reporting structure to use. The question's specification takes precedence over the "prefer later vintage" default.
- Must NOT alter arithmetic, column selection within a table, or period-to-row mapping logic (preserves semantic-entity-binding and temporal-period-anchoring).
- The gate is satisfied by ANY inspection of all provided sources and an explicit justification of vintage choice — whether the agent arrives at that justification because of this skill or voluntarily.
- Must NOT prevent combining values from structurally identical tables when no value overlap exists and the agent has verified all sources (the scan itself satisfies the gate; finding no overlap is a valid outcome).
- Must NOT affect same-file extractions or within-source comparisons.

## High-Risk Operations

These operations are forbidden without first completing the overlap inventory across all provided source files:

- **Binding an entity+period value** from any single source file when ≥2 files with different publication dates were provided, without having inspected all files for that entity+period.
- **Using a bound value in arithmetic** (sum, difference, ratio, growth rate) when the overlap inventory is incomplete.
- **Assuming the first-opened source is authoritative** for all entity+period pairs without checking whether other provided sources contain the same entity+period with different (restated) values.
- **Selecting between overlapping values** without documenting which vintage was chosen and why. The "prefer later vintage" default must be applied explicitly, not silently.
- **Treating a null scan result** (entity+period not found in other files) as a failure — finding no overlap is a valid and sufficient outcome; the gate's purpose is the scan, not the overlap.

## Regression Risks

| Risk | Mitigation |
|---|---|
| Over-triggering: agent second-guesses correct single-source extractions when the second source is provided for methodological notes or context rather than restated values | If inspection reveals no overlapping entity+period values in the second source, the gate passes without intervention. The scan itself satisfies the requirement — it adds minimal overhead because the agent only needs to search for the specific entity name near the specific period label. |
| Vintage override conflict: the "prefer later vintage" default may conflict with questions that intentionally ask for the earlier reporting structure (e.g., "before the Air Force was separated from Army") | When a question describes a specific reporting structure that is confirmed by a particular source's footnote or methodology statement, that source takes precedence over the "later is better" default. Example: if the question says "Air Force still charged to Army" and the 1948 bulletin's footnote confirms that structure, use the 1948 bulletin's value even if a later bulletin restates the figure under a different organizational structure. |
| Inspection overhead: scanning multiple large source files for every entity+period pair adds latency | The scan only requires searching for the specific entity name near the period label — a targeted grep or keyword search, not a full-table extraction. If the entity name or period label does not appear in a file, the scan for that file completes immediately. |
| Interaction with cross-vintage-structural-verification: both skills may fire when the agent needs to combine values from different vintages | This skill gates on entity+period value overlap detection and vintage authority resolution (which value to use). Cross-vintage-structural-verification gates on whether the table structures are compatible for combination. Both can fire independently: an agent might first resolve which vintage's value to use (this skill), then check whether two chosen vintages' tables are structurally equivalent for time-series construction (structural verification). The skills are complementary, not conflicting. |
| Interaction with temporal-period-anchoring: both skills may fire when periods span multiple source files | Temporal-period-anchoring gates on period-to-source mapping (which period's data is in which file). This skill gates on vintage authority within overlapping periods (which file's value to use when two files report the same period). They operate at different stages: anchoring resolves the mapping first; overlap-reconciliation then checks, for each mapped period, whether multiple sources claim a value and which one is authoritative. |
