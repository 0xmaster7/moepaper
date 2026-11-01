# CLAUDE.md — MoE malware paper

Context for Claude Code sessions in this repository. Read `pipeline_v2/RUNBOOK.md` for the full plan.

## Project in one paragraph

This is a dual-gate Mixture-of-Experts over three static-analysis experts:
- MalConv2-GCT on raw bytes
- SAFE on function semantics from radare2
- a call-graph expert (LDP by default; node2vec is the legacy ablation)

EMBER features plus availability masks drive two gates: `gate_bin` for detection and `gate_retr` for an InfoNCE-trained retrieval embedding searched with FAISS. Data is SOREL-20M malware plus self-collected benign files.

The paper was rejected by IJIS (Springer) for a weak state-of-the-art comparison and a small dataset. `pipeline_v2/` is the rebuild that answers both points. The target journals are EAAI / ESWA, then Cybersecurity (Springer), then IEEE Access.

## Layout

- Root `*.py`: the **legacy** pipeline (v1). Keep it for reference. Do not extend it. Several of its known bugs are documented in `pipeline_v2/RUNBOOK.md` §0.
- `pipeline_v2/`: everything new. Run commands from `pipeline_v2/`.
- `MalConv2/`, `safe_final/`: expert model code and weights (`safe_final/data/safe.pb` comes from `download_model.sh` or `s3://<bucket>/assets/`).

## Commands

```bash
cd pipeline_v2
python tests/smoke_test.py                      # must pass before and after any change (no AWS needed)
python extract/worker.py --max-chunks 1         # pilot one chunk (cloud box, needs radare2 + thrember)
python tools/progress.py --detail               # fleet progress / coverage / UNK rate
python build/build_dataset.py --graphs
python ablations/run_ablations.py --sweep-dropout
python ablations/run_ablations.py
python report/make_tables.py
```

## Invariants — never break these

1. **No malware on a laptop, ever.** SOREL objects are only read on EC2 in us-east-1. Never write code that downloads them to a persistent local path. `/dev/shm` for radare2 is the only on-disk copy, and it is deleted immediately.
2. **Never execute any sample.** Only static analysis (radare2, pefile, thrember, MalConv on bytes).
3. **Header parity.**
   - Every byte-level or header-level feature (MalConv, EMBER) uses `pe_parity.model_view()`.
   - radare2 uses `analysis_view()`.
   - Never feed un-normalised headers of one class only.
4. **Test is touched once.**
   - Hyper-parameters, thresholds, alpha for fusion, temperature and expert dropout are chosen on VAL.
   - `evaluate.py` already does this, so do not add test-set tuning.
5. **Splits are immutable once results exist.** `splits_*.npz` carry a dataset fingerprint, and every loader checks it. If the dataset changes, regenerate everything and say so.
6. **Benign groups never straddle splits.** `build_dataset.py` asserts this; do not remove the assert.
7. **Same feature extractor version for both classes** (thrember). Do not mix in SOREL's precomputed EMBER v2 vectors.
8. **Benign binaries stay private.** Commit hashes and features only. Never commit binaries, `data_v2/` or `results/runs/*/outputs.npz` (large).
9. Report single experts on their **covered** subsets and compare the MoE on the **same** queries (`experts_matched`).

## Tasks Claude Code can own

- **Pilot debugging on the cloud box:**
  - Run the pilot chunk.
  - Inspect raw `r2_extract.py` JSON for 5–10 files (`python common/r2_extract.py /dev/shm/x aaa 2000 150`).
  - Adapt the parser if radare2's JSON keys differ (`aflj` `offset`↔`addr`, `agCj` format).
  - Keep the SAFE normalisation identical to `safe_final/asm_embedding/FunctionAnalyzerRadare.py`.
- **Feature QA:** after Stage A, write a notebook or script that checks for:
  - UNK rate by class
  - r2 coverage by class and source
  - EMBER NaNs
  - duplicate SHA-256s
  - header-field parity: every model-view sample should have machine = subsystem = 0
- **Running and monitoring the training, ablations and baselines** (`run_ablations.py` is resumable). Summarise `results/tables.md`.
- **Extra analyses reviewers will ask for:**
  - per-source benign FPR (does any benign source get flagged more?)
  - per-tag F1 with support
  - McNemar or bootstrap CIs between the MoE and the best baseline
  - temporal drift: random vs temporal split
- **Figures** from `eval.json` files:
  - ROC on a log-FPR axis
  - risk–coverage curve
  - gate-weight heatmap by availability pattern
  - t-SNE/UMAP of `h` coloured by tag
- **Paper text:**
  - Update the LaTeX/Word tables from `results/tables.md`.
  - Rewrite the methods for the fixed SAFE tokenisation, header parity and the LDP graph expert.
  - Write the data statement (hashes and features available; binaries from the original sources).
  - Make the EAAI abstract state the AI contribution and the engineering application explicitly, with no undefined acronyms, and switch to single-column format.
  - Prepare the anonymised version (no repo link) for double-anonymous venues.
- **Code hygiene:** unit tests for `r2_extract` parsing using saved JSON fixtures (no binaries in the repo), plus type hints and docstrings.

## Tasks that need the human (Claude Code should stop and ask)

- **Anything that spends money or changes the AWS account:** creating buckets or IAM roles, quota requests, launching instances (`infra/launch_spot.sh`), Windows EC2 harvesting. Claude may prepare commands but must not run them without explicit approval.
- Requesting BODMAS / VirusTotal / ReversingLabs access (institutional identity).
- **Final decisions:**
  - the graph expert (LDP vs node2vec)
  - the target journal and submission order
  - author list and corresponding author (the IJIS submission went through the faculty co-author's account)
- Pushing to GitHub or making the repo public.

## Style

- Plain Python 3.10, numpy < 2 (see `pipeline_v2/requirements.txt`). No new heavy dependencies without a reason.
- Every script reads `pipeline_v2/config.yaml`. Override with `MOE__section__key=value` env vars, not hard-coded paths (the legacy code's `/Users/amangolani/...` paths are the anti-pattern).
- Prints should be one line per chunk or epoch. Results go to JSON under `pipeline_v2/results/`.
