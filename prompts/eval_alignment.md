# Lean4 Statement Fidelity and Equivalence Check

## Your Task

You are given an informal natural language description of a mathematical problem, a "Ground Truth" Lean4 statement (a correct formalization of this problem), and a "Generated" Lean4 statement.

Your task is to determine if the "Generated" Lean4 statement is a **faithful and mathematically equivalent formalization of the Informal Problem**, when compared against the "Ground Truth" as a reference for correct interpretation, scope, and mathematical intent.

## Input

*   **Informal Problem:** Natural language description of the mathematical problem.
*   **Ground Truth Lean4:** The reference Lean4 statement.
*   **Generated Lean4:** The Lean4 statement to be evaluated.

## Analysis Guidelines

Compare "Generated Lean4" with "Ground Truth Lean4," always relating back to the "Informal Problem."

1.  **Equivalence of Core Mathematical Assertion:**
    *   Do both Lean4 statements aim to formalize the *same fundamental mathematical fact* derived from the "Informal Problem"?
    *   Are their main propositions (the goal/body of the theorem/definition) logically equivalent in the context of the "Informal Problem"?

2.  **Fidelity to the Informal Problem (using Ground Truth as a reference):**
    *   **Variables, Types, and Scope:** Do they represent the same mathematical entities and address the same scope and generality implied by the "Informal Problem" (as captured by "Ground Truth")?
    *   **Hypotheses/Assumptions:** Are the hypotheses in "Generated" logically equivalent to those in "Ground Truth" *in terms of what they express from the Informal Problem*?
    *   **Definitions:** If custom definitions are used, do those in "Generated" serve an equivalent mathematical purpose to any in "Ground Truth" or standard library definitions, in the context of formalizing the "Informal Problem"?
    *   **Omissions/Additions:**
        *   Does "Generated" omit crucial conditions or aspects of the "Informal Problem" (which are presumably present in "Ground Truth")?
        *   Does "Generated" add conditions, complexities, or change the conclusion in ways not justified by the "Informal Problem" (when compared to "Ground Truth"'s formalization)?
    *   **Overall Relevance:** Does "Generated" accurately reflect the "Informal Problem"? If "Generated" significantly misinterprets the "Informal Problem" in a way "Ground Truth" does not, it is not a faithful formalization.

3.  **Criteria for a Faithful and Equivalent Formalization:**
    *   "Generated" must accurately formalize the "Informal Problem" by capturing the same core mathematical assertion, scope, and intent as the "Ground Truth" does.
    *   The main propositions of "Generated" and "Ground Truth" should be logically equivalent when interpreted in the context of the "Informal Problem."
    *   A "Generated" statement is NOT faithful/equivalent if:
        *   It formalizes a recognizably different mathematical problem.
        *   It addresses a significantly broader or narrower version of the problem than implied by the "Informal Problem" (and formalized by "Ground Truth").
        *   Its logical structure (hypotheses, conclusion) fundamentally misrepresents the "Informal Problem's" conditions or goal, especially when "Ground Truth" provides a correct model.

## Output Requirements

A single JSON object with the following two keys, in order:

1.  **`analysis`**: (string)
    This field should contain your detailed, step-by-step analysis performed by following the "Analysis Guidelines" above. Your analysis must:
    *   Systematically compare "Generated Lean4" with "Ground Truth Lean4," always relating both back to the "Informal Problem."
    *   Explicitly address each point outlined in the "Analysis Guidelines" (Equivalence of Core Mathematical Assertion; Fidelity to the Informal Problem including variables, types, scope, hypotheses, definitions, omissions/additions, and overall relevance).
    *   This comprehensive analysis serves as the justification for your `label`.
2.  **`label`**: (boolean)

## Example

**Informal Problem:**
"Prove that for any two natural numbers, if their sum is zero, then both numbers must be zero."

**Ground Truth Lean4 (Common for all examples below):**
```lean
def SumIsZero (a b : ℕ) : Prop := a + b = 0
def BothAreZero (a b : ℕ) : Prop := a = 0 ∧ b = 0

theorem sum_zero_implies_both_zero_gt (a b : ℕ) (h_sum_is_zero : SumIsZero a b) : BothAreZero a b :=
  sorry
```

**Generated Lean4 (Example 1 - Faithful & Equivalent):**
```lean
def MySumDefinition (n m : ℕ) : Prop := n + m = 0
def MyZeroConjunction (n m : ℕ) : Prop := n = 0 ∧ m = 0

theorem generated_sum_implies_zero_pair (x y : ℕ) (h_sum : MySumDefinition x y) : MyZeroConjunction x y :=
  sorry
```

**Example Output 1:**
```json
{
  "analysis": "Both the Ground Truth and Generated statements formalize a theorem concerning two natural numbers. The Informal Problem states that if the sum of these numbers is zero, then both numbers must be zero. \nIn Ground Truth, `SumIsZero a b` represents `a + b = 0`, and `BothAreZero a b` represents `a = 0 ∧ b = 0`. \nIn Generated, `MySumDefinition n m` (equivalent to `n + m = 0`) and `MyZeroConjunction n m` (equivalent to `n = 0 ∧ m = 0`) serve the same mathematical purpose as the definitions in Ground Truth. \nBoth statements operate on natural numbers (ℕ). The hypothesis `h_sum : MySumDefinition x y` in Generated directly corresponds to `h_sum_is_zero : SumIsZero a b` in Ground Truth. The conclusion `MyZeroConjunction x y` in Generated is mathematically equivalent to `BothAreZero a b` in Ground Truth. \nTherefore, the Generated statement accurately captures the same mathematical entities, scope, core assertion, and logical structure as the Ground Truth, reflecting the Informal Problem.",
  "label": true
}
```

**Generated Lean4 (Example 2 - Not Faithful/Equivalent):**
```lean
def ConditionSumZero (val1 val2 : ℕ) : Prop := val1 + val2 = 0
def FirstInputIsZero (val1 : ℕ) : Prop := val1 = 0 -- Note: This definition only considers the first input

theorem generated_sum_implies_first_zero (a b : ℕ) (h_condition : ConditionSumZero a b) : FirstInputIsZero a :=
  sorry
```

**Example Output 2:**
```json
{
  "analysis": "The Informal Problem requires proving that if the sum of two natural numbers is zero, then *both* numbers are zero. \nGround Truth formalizes this with `SumIsZero a b` (meaning `a + b = 0`) as the hypothesis and `BothAreZero a b` (meaning `a = 0 ∧ b = 0`) as the conclusion. \nIn Generated, `ConditionSumZero val1 val2` (meaning `val1 + val2 = 0`) is equivalent to Ground Truth's hypothesis condition. However, its conclusion `FirstInputIsZero a` translates to `a = 0`. \nWhile both statements operate on natural numbers and share an equivalent hypothesis structure regarding the sum, the Generated statement's conclusion only asserts that the first number is zero. It omits the condition that the second number must also be zero, which is a crucial part of the Informal Problem's goal and correctly captured by Ground Truth's `BothAreZero a b`.",
  "label": false
}
```

**Generated Lean4 (Example 3 - Not Faithful/Equivalent):**
```lean
def IntegerSumToZero (x y : ℤ) : Prop := x + y = 0
def AreOpposites (x y : ℤ) : Prop := x = -y

theorem generated_int_sum_implies_opposites (i j : ℤ) (h_sum : IntegerSumToZero i j) : AreOpposites i j :=
  sorry
```

**Example Output 3:**
```json
{
  "analysis": "The Informal Problem concerns natural numbers (ℕ). Ground Truth uses `ℕ` for variables `a` and `b` and proves `a = 0 ∧ b = 0` when `a + b = 0`. \nThe Generated statement, however, uses integers (ℤ) for variables `i` and `j`. Its hypothesis `IntegerSumToZero i j` is `i + j = 0` for integers, and its conclusion `AreOpposites i j` is `i = -j`. \nThe change in domain from natural numbers to integers is a significant deviation from the Informal Problem and Ground Truth. Furthermore, the property proven (`i = -j` for integers) is mathematically different from the property required by the Informal Problem for natural numbers (`a = 0 ∧ b = 0`). This alters both the scope and the core mathematical assertion relative to the Informal Problem.",
  "label": false
}
```