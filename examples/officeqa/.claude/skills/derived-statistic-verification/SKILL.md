---
name: derived-statistic-verification
description: Invoke when the question asks for a derived statistic computed using a named mathematical or statistical operation (moving average, CAGR, weighted average, geometric mean, standard deviation, percent change, growth rate, index value, quartile, Q1, Q3, percentile, IQR, Tukey method, etc.) or specifies a rounding-target precision for a computed numeric result.
---

## When To Use

The question asks for a numeric result obtained by applying a named mathematical or statistical operation to values extracted from source documents. The operation is explicitly named or strongly implied by domain terminology in the question — not merely "add" or "subtract" but a recognized statistical or derived-quantity construct.

Common activation patterns:

- "centered moving average," "simple moving average," "weighted moving average," "exponential moving average"
- "weighted average," "geometric mean," "harmonic mean"
- "compound annual growth rate" (CAGR), "annualized growth rate"
- "standard deviation" (population or sample), "variance"
- "percent change," "percentage change," "growth rate" (comparing two or more periods)
- "index value" (computed as a ratio times a base)
- "proportion," "ratio expressed as [unit]" (e.g., "expressed as a percentage," "in basis points")
- "quartile" or "percentile" computed using a named method: "Tukey inclusive," "Tukey exclusive," "Tukey hinges," "nearest-rank," "linear interpolation"
- "Q1," "Q3," "first quartile," "third quartile," "interquartile range (IQR)" computed using a named method
- Question names a quantile-based statistic (quartile, IQR, percentile) and specifies a rounding-target precision for the result
- Question specifies a rounding precision for a computed result: "rounded to the nearest [unit]," "to [N] decimal places," "rounded to the nearest thousandth"

Also activate when a question asks for a rate, ratio, or derived measure that must be computed as an arithmetic combination of multiple extracted values — even if no specific named operation appears — when the question specifies a rounding-target precision for the result. The precision-specification trigger alone is sufficient. Also activate when the question names a quantile or quartile computation method (Tukey inclusive, Tukey exclusive, Tukey hinges, nearest-rank, linear interpolation) or asks for Q1/Q3/median/IQR using a named method — the named-method trigger alone is sufficient.

## Do Not Use

- The answer is a direct lookup of a single cell value from a source table with no arithmetic combination required, and no rounding precision is specified beyond what the source already provides.
- The question asks for a simple sum or difference of exactly two values with no statistical naming and no rounding specification (e.g., "What is the total of revenue and expenses?" without "rounded to").
- The question asks for qualitative interpretation or description rather than a numeric result.
- The named operation is unambiguously a basic arithmetic operation with no statistical interpretation and no rounding specification (e.g., "add," "subtract," "total of" without precision targets).
- The precision specification refers to units conversion rather than rounding a computed result (e.g., "in millions of dollars" where values are already expressed in millions, or "in billions" as a display convention).
- The answer is a direct lookup of a single quartile/percentile value already computed in the source table (no arithmetic combination required). The question mentions "median" or "quartile" only as a descriptive label for a source column, not as a computation target where the agent must apply the named method to raw data.

## Failure Mechanism

When a question names a statistical or derived-quantity operation, agents often substitute the named operation with a simpler computation without explicit formula definition or equivalence verification. For example, "centered moving average" is replaced by "arithmetic mean" without stating that CMA(k) = arithmetic mean for k equally-spaced points — a true equivalence, but one that must be verified explicitly.

Separately, agents frequently round intermediate values (e.g., each component rate) before combining them, then round again at the final answer. This two-stage rounding discards precision and can produce small but real numeric drift versus carrying full-precision intermediates and rounding once at the end.

Together, these two failure modes — formula substitution without verification and intermediate precision loss — produce answers that are approximately correct but not precisely correct according to the computation the question actually specified.

## Procedure

This procedure confirms that the named operation was applied correctly. It is a **check**, not a recomputation. If the agent has already applied the correct formula variant (e.g., Tukey exclusive for a question that names "Tukey exclusive"), used full-precision intermediates, and rounded only at the end, **confirm the result and skip to step 6**. Do not recompute or override a correctly-computed answer. The verification detects errors; it does not substitute a new computation for an already-correct one.

1. **Identify the named operation.** Read the question and extract the exact mathematical or statistical operation it names. Record the operation verbatim. If the question does not name an operation but specifies a rounding precision for a multi-value computation, treat the implied operation as "combine extracted values arithmetically as the question directs."

2. **State the formal formula.** Write down the mathematical definition of the operation using standard notation. For example:
   - Centered moving average of window size k: `CMA(t) = (X_{t-(k-1)/2} + ... + X_t + ... + X_{t+(k-1)/2}) / k`
   - Compound annual growth rate: `CAGR = (V_final / V_initial)^(1/n) - 1`
   - Weighted average: `WA = Σ(w_i * X_i) / Σ(w_i)`
   - Percent change: `((V_new - V_old) / V_old) * 100`
   - Tukey exclusive Q1 (for odd n): sort values ascending → split into lower half and upper half excluding the median → Q1 = median of the lower half
   - Tukey exclusive Q1 (for even n): sort values ascending → split into lower half and upper half (equal halves, no median to exclude) → Q1 = median of the lower half
   - Tukey inclusive Q1: sort values ascending → split into lower half and upper half including the median in both halves → Q1 = median of the lower half
   - Q3 is computed analogously using the upper half; IQR = Q3 − Q1
   Do not skip this step — even for operations as common as "average" or "percent change." State the formula explicitly before any computation.

3. **Verify any simplification.** If the computation plan substitutes a simpler form (e.g., using arithmetic mean in place of CMA), explicitly verify the equivalence:
   - For CMA with equally-spaced points and window k: CMA(k) IS the arithmetic mean of the k points. State this equivalence.
   - If the equivalence does not hold (e.g., weighted moving average ≠ simple mean, geometric mean ≠ arithmetic mean), do not substitute. Use the correct formula.
   - If unsure whether a simplification is valid, do not simplify — use the formal formula directly.

4. **Extract raw values at full available precision.** Extract each source value exactly as it appears in the document, with all available digits. Do not round, truncate, or abbreviate at this stage. If a value is given as a fraction, ratio, or percentage, record its exact form.

5. **Compute with full-precision intermediates.** Carry all intermediate results at the full precision of the raw extracted values. If displaying intermediate steps for readability, you may show them truncated — but the actual computation chain must use the unrounded values. Do not feed a displayed (truncated) value into the next computation step.

6. **Apply rounding only at the final answer.** After the complete computation chain produces a final raw result, apply the rounding rule specified in the question:
   - "Rounded to the nearest thousandth" → round to 3 decimal places
   - "Rounded to the nearest hundredth" → round to 2 decimal places
   - "Rounded to the nearest [unit]" → round to the nearest multiple of that unit
   - If no rounding is specified, use the precision convention implied by the input data (typically the same number of significant figures or decimal places as the inputs).
   State the rounding rule and show the unrounded value before applying it.

## Invariants

- Must NOT reject a valid computation path. If the agent's chosen approach (e.g., component-wise rate computation then average, vs. aggregate numerator/denominator) is mathematically equivalent to the formal formula, the skill must accept it — as long as equivalence is verified (step 3) and precision is preserved (step 5).
- Must allow both component-wise computation (compute each rate individually, then apply the named operation) and aggregate computation (combine raw numerators and denominators, then apply the operation). Both paths are valid; the requirement is that the choice is justified and precision is preserved.
- Must NOT require infinite or unrealistic precision. Showing intermediate values with 6-8 significant figures is sufficient. The rule is: do not deliberately round or truncate the values used in the computation chain.
- Must NOT reject a formula that is trivially correct. If the stated formula matches the standard definition, confirm and proceed. Do not demand an exhaustive proof.
- Must NOT conflict with computation-trace-verification. This skill governs formula selection and precision preservation; computation-trace-verification governs arithmetic execution trace. If both activate, apply each procedure independently — do not duplicate the full trace.

## High-Risk Operations

- Rounding any intermediate value before the final answer — this is the most common precision-loss error. Each rounding step discards information that cannot be recovered.
- Substituting a named operation with a different computation without explicit equivalence verification — even when the substitution is correct (CMA(3) = mean), the verification step is mandatory.
- Assuming the named operation means "just average them" — terms like "moving average," "weighted average," and "geometric mean" are distinct operations with distinct formulas. Do not conflate them.
- Rounding the final answer to a different precision than the question specifies — verify the rounding target before applying it.
- Conflating Tukey inclusive and Tukey exclusive methods — these use different splitting rules and produce different quartile values for the same dataset. If the question names "Tukey exclusive," do not apply the inclusive rule (median in both halves). If it names "Tukey inclusive," do not exclude the median. When only "Tukey" is stated without qualification, determine which variant the source context implies; do not guess.

## Regression Risks

- Could add overhead to simple computations that do not need formula verification. Mitigated by the "named statistical operation" and "rounding specification" activation gates — simple sums and lookups without precision targets should not activate.
- Could cause over-verification of trivially correct formula choices (e.g., requiring proof that "average" means sum divided by count). The procedure requires stating the formula, not proving it from axioms.
- Could cause false negatives if the verification procedure flags a correct formula as wrong. The skill must use "verify and confirm" semantics: state the formula, check equivalence, and proceed when correct — not assume errors until proven otherwise.
- Could interact with column-scope-verification and table-title-verification on the same query. These skills govern different stages (table selection, column matching, formula definition) and should compose. Apply each independently in sequence.
