# Mathematical Problem Equivalence Prompt

## Your Task

You are given two natural language descriptions of mathematical problems, labelled "Problem A" and "Problem B". Your task is to determine if these two descriptions refer to the *same underlying mathematical problem*, even if they are worded differently or focus on slightly different aspects.

## Input

You will receive two pieces of text:

*   **Problem A:** The natural language description of the first problem.
*   **Problem B:** The natural language description of the second problem.

## Analysis Guidelines

Carefully compare Problem A and Problem B. Determine if they are fundamentally asking the same mathematical question by considering points like these:

1.  **Mathematical Objects & Structures:**
    *   What are the main mathematical concepts or structures involved (e.g., graphs, sets, numbers, functions, permutations, geometric figures)?
    *   Are these structures essentially equivalent in both problems, even if named differently (e.g., "points and connections" vs. "vertices and edges")?

2.  **Conditions & Constraints:**
    *   What rules, properties, assumptions, or limitations are placed on the objects or their relationships (e.g., size constraints, adjacency rules, algebraic properties, ordering)?
    *   Are these sets of conditions logically equivalent? Do they restrict the problem space in the same way?

3.  **Goal or Question:**
    *   What is the primary objective of each problem?
    *   Are the core goals functionally identical?

4.  **Equivalence under Reformulation:**
    *   Could one problem be transformed into the other through simple relabeling of variables or terms?
    *   Could one be restated using slightly different mathematical language or perspective (e.g., focusing on a complementary problem, using duality) without changing the essential challenge?

5.  **Scope and Parameters:**
    *   Do the problems operate at the same level of generality?
    *   Are the key parameters (like the size `n`, specific constants, etc.) playing equivalent roles?

6.  **Extremal Cases & Boundaries:**
    *   If the problems involve optimization or bounds, are they concerned with the same extremal properties (e.g., maximum independent set size, minimum coloring number)?

7.  **Implicit Information & Context:**
    *   Consider the typical mathematical context (e.g., combinatorics, number theory, algebra). Do both problems fit naturally within the same context and rely on similar background knowledge?

## Output Requirements

Your response must be a single JSON object containing exactly two keys:

1.  **`label`**: A boolean value.
    *   Set to `true` if the problems describe the same mathematical challenge.
    *   Set to `false` if the problems describe different mathematical challenges.
2.  **`justification`**: A string containing a concise explanation supporting your conclusion. Briefly highlight the key mathematical similarities or differences based on your analysis (objects, constraints, goals, etc.). Focus on the *mathematical substance*, not superficial wording variations unless they imply a mathematical difference. Avoid simply repeating the problem statements.

Ensure the output is valid JSON.

## Example

**Problem A:**
A school organizes its students into five different study groups. Each student belongs to exactly one group. Over time, some students have collaborated on special projects, and it's known that any two students in the school have either worked on a project together or are in the same group. However, if three students have all worked with each other on projects, then at least two of them must come from the same study group. Given these conditions, show that there must be at least one study group with five or more students.

**Problem B:**
Let \( S \) be a finite set of students partitioned into five disjoint groups \( G_1, G_2, G_3, G_4, G_5 \). Define a graph on the vertex set \( S \), where an edge connects two students if they have worked together on a project. Assume the following holds: for every pair of distinct students \( a, b \in S \), either \( a \) and \( b \) belong to the same group, or they are connected by an edge in the graph. Furthermore, no three students who form a triangle in the graph belong to three different groups. Prove that at least one group \( G_i \) contains at least five students.

**Example Output:**

```json
{
  "label": true,
  "justification": "Both problems describe the same mathematical setup: a set of students partitioned into 5 groups and a collaboration relationship defined between students. Problem B uses explicit graph terminology (vertices, edges, triangles) to represent the collaboration structure described narratively in Problem A. The core constraints are identical: (1) any two students are either in the same group or collaborated, and (2) no set of three collaborating students comes from three distinct groups. The goal is also identical: to prove that at least one group must have a size of 5 or more. The underlying mathematical structure, constraints, and objective are equivalent."
}
```