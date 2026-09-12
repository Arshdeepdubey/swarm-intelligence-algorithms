# Documentation Review: IEEE Paper vs. SDP Report vs. Evaluation Slides vs. Implementation

This document is the requested cross-check between the three source
documents that were uploaded (`docs/source-documents/`) and the code that
now implements them (`src/swarm_fs/`). It records what is consistent,
what is missing or ambiguous in the source documents, and what the actual
(now-executable, now-tested) implementation had to assume to fill those
gaps.

## 1. Documents reviewed

| Document | Role | Length |
|---|---|---|
| `IEEE_Conference_Paper.docx` | Broad literature survey: "A survey of swarm intelligence-based feature selection algorithms" | ~6 sections, 25 references |
| `SDP_Final_Project_Report.docx` | The team's actual Senior Design Project report — bundles the university's generic report-formatting guidelines *and* the completed report for "A Study on Swarm Intelligence Algorithms for Optimal Feature Selection" in one file | 8 tables, 11 inline images, ~50 pages per its own table of contents |
| `Preliminary_SDP_Evaluation.pptx` | 14-slide preliminary evaluation deck presented to the panel | 14 slides |

## 2. Cross-document consistency (what checks out)

- **Algorithm scope matches between the report and the slide deck.** Both
  the SDP report (Sec 1.1, "Existing System") and the PPT (Slides 7-10)
  cover exactly four algorithms — PSO, ACO, ABC, CSA — with the same
  equations reproduced verbatim in both:
  - PSO velocity/position update (report Sec 3.3 Fig. 3 = PPT Slide 7)
  - ACO edge-selection + global pheromone update (report Sec 3.3 = PPT
    Slide 8)
  - ABC employed-bee/roulette-wheel/scout-bee equations (report Sec 3.3 =
    PPT Slide 9)
  - CSA Levy-flight step equation (report Sec 3.3 = PPT Slide 10)

  All four are implemented with these exact equations in
  `src/swarm_fs/algorithms/{pso,aco,abc,cuckoo}.py` (each module's
  docstring quotes the specific equation it implements).

- **Dataset description matches between report text and report Table 4.**
  Iris (150 x 4), Breast Cancer (569 x 30), Wine (178 x 13), Digits
  (1,797 x 64) are consistent between the narrative text (Sec 3.1) and
  Table 4, and both match `sklearn.datasets`' actual shapes — verified
  directly in `tests/test_datasets.py::test_dataset_shapes_match_sdp_report_table1`.

- **The fitness formulation is unambiguous and directly implementable.**
  The report's plain-English description of `SVMFeatureSelection`
  (Sec 3.3: "select features based on binary values... if no features
  selected, return default value... train SVM and calculate accuracy...
  weighted score") maps one-to-one onto
  `src/swarm_fs/fitness.py::SVMFeatureSelection`.

- **Team/attribution details are consistent** across the report's cover
  page, certificate, and the PPT's title slide (same four names,
  registration numbers, supervisor, group number L2).

## 3. Gaps in the source documents (and how the implementation resolved them)

These are the most important findings of this review — cases where the
report/paper describe *what* the algorithm does but not the exact
parameter values needed to run it, because the report's own parameter
table and results tables were embedded as **images**, not extractable
text/data:

| Missing parameter | Where it would appear | Value used in this implementation | Source |
|---|---|---|---|
| Population size | Table 7 ("Parameters Used") | **10** | Actually present in Table 7 — this one *is* documented |
| Iterations | Table 7 | **100** | Actually present in Table 7 |
| Search bounds | Table 7 | **[0, 1]** | Actually present in Table 7 |
| PSO inertia weight `w`, `c1`, `c2` | Not tabulated anywhere | `w=0.7`, `c1=c2=1.5` | Standard literature defaults (Kennedy & Eberhart-style) |
| ACO evaporation rate `rho`, `alpha`, `beta`, `Q` | Not tabulated anywhere | `rho=0.5`, `alpha=beta=1`, `Q=1.0` | Common ACO defaults |
| ABC scout "limit" | Not tabulated anywhere | `population_size * dim` (floor 10) | Standard Karaboga-style heuristic |
| CSA discovery probability `pa` | Not tabulated anywhere | `0.25` | Yang & Deb's original paper |
| SVM kernel/`C` | Not specified | linear kernel, `C=1.0` | Reasonable default; report never names a kernel |
| Fitness weight `alpha` (accuracy vs. feature-count) | Described qualitatively ("weighted score") but never given a number | `0.99` | Chosen to heavily favor accuracy, matching the report's framing that accuracy is primary and feature-count is a tie-breaker |
| Exact experimental result numbers (Sec 4.3, "Experimental Outcomes") | Referenced as images/screenshots in the report, not machine-readable text or tables | N/A — re-run from scratch | See Section 4 below |

Every one of these assumptions is also documented as a code comment/
docstring at its point of use, so a reader auditing the implementation
against the report doesn't have to cross-reference this file.

**Recommendation for the next report revision:** replace the embedded
result screenshots and the (currently very sparse) Table 7 with a
machine-readable appendix — even a simple key-value list of every
hyperparameter used, per algorithm — so the study is exactly
reproducible from the report text alone.

## 4. Reproduced results vs. the report's stated conclusion

The report's conclusion (Sec 5) states: *"in comparison with the table we
got to know that PSO is performing better as compared to the other three
algorithms."* Because the report's own results tables are images (not
extracted text — see Section 3), this repository's `results/demo_results.csv`
is an **independent re-run**, not a byte-for-byte reproduction, using the
one set of parameters the report does document (population=10,
iterations=100) plus the literature-default parameters listed above.

The independent run (160 total runs: 4 algorithms x 4 datasets x 10 runs,
full detail in `results/README.md`) shows a **more mixed picture** than
the report's single-winner conclusion:

| Dataset | Best algorithm (this run) | Margin over 2nd place |
|---|---|---|
| Iris | ABC / ACO / Cuckoo (three-way tie) | PSO ~0.2 pp behind |
| Wine | PSO | ~0.4 pp over ACO |
| Breast Cancer | ACO (tied with PSO) | ~0.2 pp over ABC |
| Digits | ABC (tied with PSO) | ~0.02 pp over Cuckoo |

PSO is competitive everywhere (never worse than 2nd-3rd place) but is not
the outright best on 3 of 4 datasets in this run — a plausible outcome
given none of PSO's or the other algorithms' fine-tuning parameters were
specified in the source report, so this run and the report's original run
are not using identical hyperparameters. This is flagged as a
discrepancy rather than a correction: with the original run's exact
parameters unavailable, neither result invalidates the other, but it does
mean the report's specific "PSO wins" claim should be read as
run-dependent rather than a robust, reproducible finding, pending the
report being revised per the recommendation in Section 3.

## 5. Scope differences between the survey paper and the implementation

The IEEE survey paper (Sec 4.4, "Remaining Algorithms") also discusses
Glowworm Swarm Optimization (GSO), the Bat Algorithm (BA), and the
Firefly Algorithm (FA) as SI-based feature selection methods in the wider
literature. The SDP report and PPT scope the actual implementation down
to four algorithms (PSO, ACO, ABC, CSA) and this repository matches that
narrower, *implemented* scope — GSO/BA/FA are cited as related work in
`docs/literature_review.md` but have no corresponding module in
`src/swarm_fs/algorithms/`. This is a **correct, intentional scope
match** between the report and the code, not a gap — flagged here only
so a reader of the broader survey paper isn't surprised not to find
`gso.py`/`bat.py`/`firefly.py`.

## 6. Report-format compliance (`Revised_SDP_Final_Project_Report_Format.docx`)

The bundled report-format guidelines specify a required CD structure
accompanying the bound report:

```
Presentation / Documentation / Source Code / Program / Support / Help
```

This repository maps directly onto that structure (adapted for a
GitHub-hosted research codebase rather than a physical CD):

| Required folder | This repository |
|---|---|
| Presentation | `docs/source-documents/Preliminary_SDP_Evaluation.pptx` |
| Documentation | `docs/source-documents/*.docx`, `docs/literature_review.md`, this file |
| Source Code | `src/swarm_fs/`, `scripts/`, `tests/` |
| Program | N/A — this is a research library, not a standalone executable; `scripts/run_experiments.py` is the closest equivalent (a runnable entry point) |
| Support | `requirements.txt` / `pyproject.toml` (declares every third-party library used, in place of a runtime installer) |
| Help | `README.md` Quickstart section |

One explicit formatting rule is worth flagging: the guidelines state the
*bound report itself* "must not contain any description of... technology
or platform or OS or tools used... without including any source code."
That constraint applies to the physical thesis document, not to a
software engineering repository — so `docs/literature_review.md` and this
review intentionally *do* reference specific modules and functions, which
is appropriate practice for a codebase but would need to be stripped back
out if this content were pasted into the bound report proper.

## 7. Minor observations

- The SDP report's own reference list (Sec 6, 7 entries — 3 of which are
  plain Wikipedia links) is noticeably thinner than the IEEE survey
  paper's 25 peer-reviewed references. Since the report's Sec 1.1
  ("Existing System") draws directly on concepts from the survey paper,
  a future revision could strengthen the report's bibliography by pulling
  in the relevant subset of the survey paper's references (e.g. the
  original PSO/ABC/ACO/CSA papers already cited here in
  `docs/literature_review.md`).
- Table 2 ("List of Figures") and Table 3 ("List of Tables") in the report
  template contain stale placeholder rows (e.g. "Impact of Social
  Distancing", "Use Case Diagram") left over from a different project's
  template instance — harmless, but worth cleaning up before final
  submission.
- The individual-contributions table (report, unlabeled table before
  the Table of Contents) is a good practice worth preserving: it
  explicitly attributes literature survey, problem formulation,
  experimentation, and documentation work across all four team members.
