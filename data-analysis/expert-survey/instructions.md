# Instructions for Scoring CCCP Metrics

## Expert task: assigning Plan Depth and Maximum Plan Interactivity to 13 code snippets

---

## 1. What this study is about

We are testing whether a code-complexity metric grounded in **Cognitive Load Theory** actually predicts the cognitive effort that people experience when reading code.

The study has two halves:

- **The human side (already collected / being collected).** Students are shown 13 short code snippets and asked to predict the output of each. After each snippet they report how much *cognitive effort* it took to understand the code. This gives us a measure of *actual perceived load*.
- **The metric side (your job).** For the same 13 snippets we want a *theory-based prediction* of complexity, using the **Cognitive Complexity of Computer Programs (CCCP)** framework from Duran, Sorva & Leite (2018). CCCP produces two numbers per program: **plan depth (PD)** and **maximum plan interactivity (MPI)**.

Once we have both sides, we compute the correlation between the CCCP metrics and the students' reported effort. The CCCP has no automated tool, so the metrics must be assigned by hand by experts who understand the framework — that is what we are asking you to do.

You do **not** need to run the snippets, predict their output, or judge how "hard" they feel to you. You only need to apply the CCCP framework and report two scores per snippet, plus a short justification.

---

## 2. What you need to do (overview)

1. **Read the essential parts of the paper** (Section 3 below tells you exactly which parts).
2. **For each of the 13 snippets**, perform:
   - a **hierarchical analysis** → produces the **plan depth (PD)** score;
   - an **interactivity analysis** → produces the **maximum plan interactivity (MPI)** score.
3. **Record** the two scores and a brief justification for each snippet in the reporting table (Section 7).
4. When scoring MPI, **assume the reader is a first-year beginner** and judge for yourself which schemas such a student plausibly has (see the boxed note in Section 6.2) — this strongly affects MPI.

The 13 snippets (available at [https://people.inf.ethz.ch/~sverrirt/cccp/snippets/](https://people.inf.ethz.ch/~sverrirt/cccp/snippets/)) are short, single-purpose Python programs:

| # | Snippet | # | Snippet |
|---|---|---|---|
| 01 | Average of an array | 08 | Largest of three numbers |
| 02 | Decimal to binary | 09 | Median of an array |
| 03 | Exponentiation | 10 | Prime check |
| 04 | Factorial | 11 | Reverse an array |
| 05 | Intersection of circles | 12 | Sum of three digits |
| 06 | Intersection of rectangles | 13 | Swap variables |
| 07 | Largest number in an array |  |  |

A note on judgment: the framework's authors are explicit that CCCP is **not** a fully mechanical algorithm — analyzing a program still requires interpretation when you delimit plans. Treat the descriptions and worked examples below as your reference, and where a snippet is genuinely ambiguous, make the most reasonable interpretation you can and write a short note explaining the choice you made. You are encouraged to use whatever helps you understand the framework and the snippets better — searching the web, using AI tools, and running the code snippets to see what they do are all fine.

---

## 3. Essential parts of the paper

You can get everything you need from the following, in this order. Other sections (Introduction, Related Work, Contributions/Limitations, References) are useful context but **not required** for scoring.

**Must read carefully:**
- **Section 2.4 — The Model of Hierarchical Complexity (MHC).** The three rules (prerequisite, chain, **coordination**) and the complexity formula. This is the engine behind plan depth.
- **Section 3 (intro) — The Cognitive Complexity of Computer Programs.** What "plans" are and how the MHC is mapped onto programs.
- **Section 3.1 — Hierarchical Analysis**, including **Case Study 1 (Summing program)** and **Case Study 2 (Averaging program)**. This is the procedure for **plan depth**. Study Figures 1 and 2.
- **Section 3.2 — Interactivity Analysis**, including **Case Study 1 revisited**, **Case Study 2 revisited**, and **Case Study 3 (Rainfall, merged vs. sequenced)**. This is the procedure for **MPI**. Study Figures 4 and 5.

**Helpful background (skim):**
- **Section 2.1 — Schemas and Cognitive Load** (working memory, schemas, element interactivity).
- **Section 2.2 — Plans: Schemas in Programs** (what Soloway's "plans" are).
- **Section 2.3 — Complexity vs. Difficulty** (CCCP measures *complexity*, not *difficulty*).

The remainder of this document summarizes these essential parts so you have a self-contained reference. Even so, **we strongly encourage you to work through all three case studies in the paper (Case Studies 1, 2 and 3 in Sections 3.1–3.2) before you start scoring.** Following the worked plan trees and the interactivity traces end to end is the fastest way to get a feel for how plans are delimited and how the metrics behave in practice.

---

## 4. Core idea and vocabulary

CCCP describes a program in terms of the **plans** it contains. A *plan* is a stereotypical, reusable solution pattern in code — essentially a *schema* in the programmer's mind realized as concrete code (e.g., "initialize a variable", "loop over a range", "accumulate a running total"). The framework assumes a direct mapping between a plan in the code and a schema in the head of someone who understands that code.

A program is analyzed as a **tree of plans**:
- the **leaves** are *primary plans* — primitive operations of the underlying machine that don't break down further in a meaningful way;
- the **root** is the plan that corresponds to the whole program;
- intermediate nodes are higher-level plans built out of lower-level ones.

CCCP produces two numbers from this:
- **Plan Depth (PD)** — how *deep* the plan tree is (overall structural complexity of the schemas needed to reason about the program). Comes from **hierarchical analysis**.
- **Maximum Plan Interactivity (MPI)** — the largest number of plans that must be held in mind *simultaneously* at any single point while processing the program. Comes from **interactivity analysis**.

Keep these distinct: **PD is about how tall the hierarchy is; MPI is about how many plans collide at the worst moment.** A program can be deep but have low interactivity (well-compartmentalized), or shallow but highly interactive (everything tangled together).

---

## 5. Metric 1 — Plan Depth (Hierarchical Analysis)

### 5.1 The three MHC rules

When one plan is built from others, the relationship is one of three kinds. Only **one** of them increases depth.

- **Prerequisite rule.** Plan A simply requires that one other plan (at the *same* level) be in place first. This is a dependency, **not** an increase in complexity. *Does not raise depth.* (Example from the paper: "define literals" is a prerequisite for "declare variable".)
- **Chain rule.** A higher plan combines two or more lower plans whose order is **arbitrary** — the whole is just the sum of the parts (e.g., computing `1 + 2 − 4`, where addition and subtraction can be done in either order). *Does not raise depth.* The authors **ignore chaining** entirely to keep trees simple, and so should you.
- **Coordination rule.** A higher plan organizes **two or more** lower plans in a **non-arbitrary** way: each lower plan plays a **distinct role** and they **cannot be swapped or reordered**. *This is the only relationship that raises depth.* (Example: `2 × (5 + 3) = (2 × 5) + (2 × 3)` — addition and multiplication play distinct, non-interchangeable roles.)

**Litmus test for coordination:** "Do these sub-plans play different, fixed roles that can't be freely reordered or interchanged?" If yes → coordination → depth increases. If they're just done in some order with no fixed roles → chain → no increase. If it's a single one-to-one dependency → prerequisite → no increase.

### 5.2 The depth formula

- Every **primary** plan A₀ (a leaf) has depth **0**.
- A plan A that **coordinates** lower-level plans A₁ … Aⱼ has:

  **PD(A) = max( PD(A₁), …, PD(Aⱼ) ) + 1**

In words: a coordinating plan sits **one level above the deepest plan it coordinates**. Prerequisites and chains add nothing.

The **plan depth of the whole program = the depth of the root plan** (the plan representing the entire program).

### 5.3 Procedure for one snippet

1. **Find the primary plans (depth 0).** Identify the primitive machine operations the code uses — e.g., *define a literal*, *declare a variable*, *arithmetic operator*, *jump to another part of the code (control transfer)*. These are the leaves; they coordinate nothing. (Do not go below the level needed to understand the program — e.g., bit-level operations are unnecessary for these snippets.)
2. **Build upward.** For each higher-level plan, decide which lower plans it **coordinates** (distinct roles, non-arbitrary order). Mark prerequisites and chains, but remember they do not add depth.
3. **Assign each plan its depth** using PD = max(children) + 1 for coordinating plans.
4. **Identify the root** (the whole-program plan) and read off its depth. **That number is the snippet's PD.**

### 5.4 Worked example (Paper Case Study 1 — Summing program)

```c
int i, input, sum;
sum = 0;
for (i = 1; i <= 10; i++) {
  read(input);
  sum = sum + input; }
```

| Plan | Built from | Depth |
|---|---|---|
| Define literals (P1) | — (primary) | 0 |
| Declare variable (P2) | — (primary; P1 is a *prerequisite*) | 0 |
| Arithmetic operator (P3) | — (primary) | 0 |
| Jump to code (P4) | — (primary) | 0 |
| Initialize a variable (P5) | coordinates P1, P2 | max(0,0)+1 = **1** |
| Evaluate an expression (P6) | coordinates P2, P3 | max(0,0)+1 = **1** |
| Read input (P7) | coordinates P5, P2 | max(1,0)+1 = **2** |
| Accumulate in a variable (P8) | coordinates P5, P6 | max(1,1)+1 = **2** |
| Test for termination (P9) | coordinates P6, P4 | max(1,0)+1 = **2** |
| Loop over a range (P10) | coordinates P5, P9, P8 | max(1,2,2)+1 = **3** |
| **Summing program (P11)** | coordinates P10, P7, P8 | max(3,2,2)+1 = **4** |

**Plan depth of the program = 4** (the depth of the root, P11).

For a second, more complex worked example (a program with a user-defined function, reaching PD = 6), see **Case Study 2 / Figure 3** in the paper.

---

## 6. Metric 2 — Maximum Plan Interactivity (Interactivity Analysis)

### 6.1 The idea

Plan depth ignores what working memory actually has to juggle *at the same time*. Interactivity analysis fixes that. It asks: as you mentally execute the program, **how many plans must be active in working memory simultaneously**, and what is the **worst** (largest) such moment?

Two concepts from working-memory research are used:
- **Focus of Attention (FoA):** the single plan you are processing right now.
- **Region of Direct Access (RDA):** the set of plans that must be held active *alongside* the FoA to process it. (The RDA includes the focused plan itself.)

As you trace execution, the FoA moves from plan to plan and the RDA is rearranged. At each moment, the **Plan Interactivity (PI)** = the number of plans simultaneously active in the RDA. The **Maximum Plan Interactivity (MPI)** = the **largest PI** reached anywhere in the trace. **MPI is the number you report.**

### 6.2 Prior knowledge: assume a first-year beginner

> ⚠️ **The reader to model is a first-year beginner.**
> Unlike plan depth, interactivity depends on the reader's prior knowledge: a plan the reader can treat as a single familiar chunk counts as **one** element, and its internal sub-plans are abstracted away (not counted). The students in this study are **all first-year students — i.e., beginners** — so that is the prior-knowledge level you should assume.
>
> The framework does not hand you a list of which plans a beginner has already chunked, so **this is the main judgment call you make for MPI**: given a first-year beginner, decide for each plan whether such a student would plausibly recognize it as a single familiar chunk, or whether they would still have to process its sub-plans element by element. A plan the beginner has *not* yet chunked contributes its constituent sub-plans to the count, not a single element. (Contrast this with the paper's case studies, which assume an experienced reader who can chunk every major high-level plan; for beginners you should generally expect *less* chunking, and therefore often higher interactivity.)
>
> Because you are working asynchronously, you will not align this judgment with other raters in advance — that is expected. Apply your reading of "what a first-year beginner has chunked" **consistently across all 13 snippets**, and **write down the assumption you used** (which plans you treated as already-chunked) at the top of your score sheet so the research team can interpret your scores.

### 6.3 When is a plan in the RDA of the current FoA?

Include a plan A in the RDA of the focused plan B if **either**:
- A is **nested** within B, or B is nested within A; **or**
- the execution of A **interleaves** with that of B (their control flow or data flow are intertwined — e.g., they share a loop, or share a variable that ties their steps together).

Do **not** include A if its execution **finishes before** B's begins (and they don't share data that must be held). Such plans are processed at separate times and do not need to coexist in working memory.

This is why **plan composition strategy matters so much for MPI**:
- **Merged / interleaved plans** (everything happening inside one loop, sharing variables) force many plans to be active at once → **high MPI**.
- **Sequenced / compartmentalized plans** (e.g., split into separate functions, each computed, stored, then its result reused) let the reader process one plan, collapse it to a result, and move on → **low MPI**. The paper calls this a *switch–process–store–output (SPSO)* pattern.

### 6.4 Procedure for one snippet

1. Start from the snippet's code and the plan tree you built in Section 5.
2. Apply the first-year-beginner prior-knowledge assumption (Section 6.2): decide which high-level plans such a student would already chunk into a single element (abstract those away) and which they would still process as separate sub-plans (count those).
3. **Trace execution step by step.** At each step, identify the FoA (the plan being processed).
4. Build the RDA for that FoA: every plan that is nested with it or whose execution interleaves with it (shared control or data flow). **Count the plans in the RDA → that is the PI for this moment.**
5. Repeat for every FoA shift across the whole program.
6. **MPI = the maximum PI** observed. **That number is the snippet's MPI.**

### 6.5 Worked example (from the paper)

**Case Study 1 — Summing program → MPI = 3.** Three high-level plans (loop, accumulate, read input) all run inside the same loop and share the input variable. Their control and data flow are interleaved, so processing any one of them requires all three to be active together. Worst moment holds 3 plans → **MPI = 3**.

For an example of how *plan composition* drives MPI — the same problem yielding a very different MPI when written as merged plans versus sequenced plans — see **Case Study 3 (Rainfall, Figures 4 and 5)** in the paper.

---

## 7. Reporting your scores

For each snippet, fill in the table below (or a copy of it). Keep justifications short — one or two sentences plus, ideally, the root plan's depth derivation and the worst-case RDA.

**Prior-knowledge assumption used (write once, applies to all your snippets):**
Reader = first-year beginner. List which plans you decided such a student has already chunked into single elements (these are abstracted away) and which they would still process as separate sub-plans.
_e.g., "Assumed a first-year beginner can chunk: simple variable assignment and arithmetic expressions. Assumed NOT yet chunked: loops, accumulation, array indexing, library calls (sum/len/sort/sqrt) — their sub-plans are counted."_

| # | Snippet | Plan Depth (PD) | Max Plan Interactivity (MPI) | Brief justification |
|---|---|---|---|---|
| 01 | Average of an array | | | |
| 02 | Decimal to binary | | | |
| 03 | Exponentiation | | | |
| 04 | Factorial | | | |
| 05 | Intersection of circles | | | |
| 06 | Intersection of rectangles | | | |
| 07 | Largest number in an array | | | |
| 08 | Largest of three numbers | | | |
| 09 | Median of an array | | | |
| 10 | Prime check | | | |
| 11 | Reverse an array | | | |
| 12 | Sum of three digits | | | |
| 13 | Swap variables | | | |

If helpful, attach your plan tree sketch for any snippet you found ambiguous.

---

## 8. Quick reference / common pitfalls

- **PD is a max-plus-one, not a sum.** A coordinating plan is one level above its *deepest* child, regardless of how many children it has. More children ≠ more depth.
- **Only coordination raises depth.** Prerequisites (single same-level dependency) and chains (arbitrary-order combinations) do not. When in doubt, ask whether the sub-plans have *distinct, non-interchangeable roles*.
- **Ignore chaining** in the tree, exactly as the paper does.
- **Don't go below the notional machine.** Stop at primitive operations needed to understand the program (store a value, arithmetic, control transfer). Bit-level detail is unnecessary here.
- **MPI counts simultaneous plans, including the one in focus.** "Plan A nested in plan B, both active" = PI of 2, not 1.
- **Sequential ≠ simultaneous.** A plan that finishes before another starts is not in its RDA — unless they share data that must be carried forward.
- **Model a first-year beginner for MPI**, and apply your reading of what such a student has chunked **consistently across all 13 snippets**. Expect beginners to chunk *less* than the experienced reader in the paper's case studies, which tends to push interactivity up. Write down the assumption you used.
- **You are measuring complexity, not difficulty.** Do not factor in how hard *you* personally find the snippet, how it's formatted, or how unfamiliar the task is. Apply the framework as written.
