# pipeline_v2 runbook

Everything needed to rebuild the dataset at 50k–100k samples on AWS without
storing malware (or any binaries) on your laptop, retrain the MoE, run every
baseline and ablation, and produce the paper tables.

Run every command from `pipeline_v2/` unless stated otherwise.

---

## 0. What changed vs the current repo, and why

Reading the repo turned up four problems that matter more than dataset size.
Each one is fixed in this pipeline.

| # | Problem in the current code | Effect | Fix here |
|---|---|---|---|
| 1 | `safe_final/embed_text.py` builds tokens like `mov ebp, esp`, but SAFE's vocabulary keys are `X_mov_ebp,_esp`. `InstructionsConverter` maps every token to `X_UNK`. | If your SAFE embeddings came from `embed_text.py` (`batch_embed.sh` calls it), the "SAFE" expert never saw an instruction. Every function becomes a run of identical UNK ids, so the embedding only encodes function length. `tests/smoke_test.py` reproduces the mismatch against your `word2id.json`. | `common/r2_extract.py` uses SAFE's own normalisation (`filter_memory_references`) on radare2's structured JSON (`aoj`). The UNK rate is logged per sample; aim for well under 0.3. |
| 2 | SOREL malware is "disarmed": `FileHeader.Machine` and `OptionalHeader.Subsystem` are zeroed. Your benign files are intact. | MalConv (raw bytes) and EMBER header features can separate the classes from two header fields alone. radare2 also can't tell the architecture when `Machine = 0`, which likely explains part of the low SAFE/graph coverage on malware. | `common/pe_parity.py`: the model view zeroes both fields for every file. The analysis view restores `Machine` from the PE32/PE32+ magic for radare2 only. |
| 3 | node2vec is fitted separately per binary, then mean-pooled. | Each binary's vectors live in their own random rotation, so cosine similarity across binaries is not well defined. A reviewer who knows node2vec will flag it. | New default graph expert, **LDP**: local-degree-profile histograms plus call-graph statistics plus the instruction-category mix. It is deterministic, needs no fitting, and is comparable across binaries. node2vec is still computed, so it can be reported as an ablation (`abl_graph_node2vec`). |
| 4 | Raw EMBER values (sizes and counts spanning ~10 orders of magnitude) are fed straight into the gate MLP. | This contributed to gate saturation. Bounded logits treated the symptom. | `sign·log1p` followed by z-scoring with train statistics. |

Also:
- **MalConv2:** you already use it. `MalConv2/malconvGCT_nocat.checkpoint` is the authors' MalConv-GCT (Raff et al., AAAI 2021), so nothing needs switching. Name it "MalConv2 (GCT)" in the paper. Its own malware head is added as a zero-shot baseline.
- **Feature extraction:** EMBER features are re-extracted with one extractor (thrember, the EMBER2024 feature version), the same version for both classes. Mixing SOREL's EMBER-v2 vectors with self-extracted benign vectors would create a version artefact.

---

## 1. The mental model (S3 + spot)

- **S3 is a disk in the cloud.** SOREL's 10M malware binaries already sit in `s3://sorel-20m/` in `us-east-1`. You never download them. A machine in the same region reads them over AWS's internal network. That is fast, and in-region S3→EC2 traffic carries no transfer charge.
- **Compute goes to the data.** Every stage runs on an EC2 instance in `us-east-1`. Each file is streamed into RAM, processed, and dropped. Only small feature vectors are written back to your own bucket. radare2 needs a file path, so each binary exists for a few seconds in `/dev/shm`, which is RAM and not disk.
- **Spot instances** are spare EC2 capacity at a large discount. AWS can reclaim one with a **2-minute warning**. The pipeline is built for that:
  - Work is split into **chunks** of 500 samples (`manifests/chunks/chunk_XXXXX.parquet`).
  - A worker **claims** a chunk with a lease object (an S3 conditional write, so two machines never take the same chunk).
  - When a chunk finishes, its output parquet appears in `v2/features/`. A chunk with output is never redone.
  - The worker polls the instance metadata for the interruption notice. When it arrives, the worker releases its lease and stops. Any lease older than 4 hours is treated as dead and stolen.
  - **Recovery = run the launch command again.** Nothing is lost except the chunk in flight.
- Instances **terminate themselves** when there is no work left, so they don't bill you while idle.

```
SOREL S3 (malware, zlib)  ─┐
your S3 benign/raw/...    ─┼─► Stage A (CPU spot fleet): parity → EMBER, MalConv2, radare2 → SAFE tokens, call graph, LDP, node2vec
                           │        └─► s3://bucket/v2/features/chunk_*.parquet
                           └─► Stage B (CPU/GPU spot): SAFE embeddings from tokens → v2/emb_safe/
build_dataset.py ─► data_v2/dataset.npz (+ graphs.npz) ─► train / eval / baselines / tables (laptop or cloud)
```

---

## 2. One-time AWS setup (≈30 min)

1. **Set a budget alarm first.** Billing → Budgets → monthly cost budget (e.g. $50) with email alerts at 50/80/100%.
2. **Create the bucket** in `us-east-1`. Name it, for example, `moe-paper-<yourname>`, and keep "Block all public access" on. Put the name in `config.yaml` under `s3.bucket`, or `export MOE__s3__bucket=...`.
3. **Create an IAM role for EC2** named `moe-worker`:
   - IAM → Roles → Create → AWS service: EC2.
   - Attach `AmazonSSMManagedInstanceCore`. This lets you open a shell in the browser with no SSH keys and no open ports.
   - Add an inline policy from `infra/iam-policy.json`, with your bucket name filled in.
   - The console also creates the instance profile with the same name.
4. **Check the spot vCPU quota.** Service Quotas → EC2 → "All Standard (A, C, D, H, I, M, R, T, Z) Spot Instance Requests". New accounts are often at 5–32 vCPU. Request 128 and wait for approval (usually hours).
5. **Configure the AWS CLI on your laptop.** Run `aws configure`, set the region to `us-east-1`, and use an IAM user with admin rights (or EC2 + S3 + SSM).
6. **Push this folder to your GitHub repository.** The instances `git clone` it, so a public repo is simplest. A private repo works if you add a deploy token to the URL.
7. **Upload the SAFE weights once.** Run `cd safe_final && ./download_model.sh`, then `aws s3 cp data/safe.pb s3://$MOE_BUCKET/assets/safe.pb`.

```bash
export MOE_BUCKET=moe-paper-yourname
export MOE_REPO=https://github.com/<you>/moepaper.git
export MOE_INSTANCE_PROFILE=moe-worker
```

---

## 3. Collect the data

### 3.1 Get a shell in the cloud

```bash
./infra/launch_spot.sh idle 1
```

Then in the console go to EC2 → the instance → Connect → Session Manager, and run:

```bash
sudo -i; cd /opt/moe/repo/pipeline_v2; source /opt/moe/venv/bin/activate
export MOE__s3__bucket=$YOUR_BUCKET
```

Bootstrap takes about 10 minutes (it compiles radare2). Follow it with `tail -f /var/log/moe-bootstrap.log`.

### 3.2 Malware manifest (SOREL)

```bash
python data/00_sorel_manifest.py --inspect               # confirms meta.db columns
python data/00_sorel_manifest.py --include-legacy ../MoE_project_data/sha_list.txt
```

- This draws 50k malware samples across SOREL's own time partitions. Rare behaviour tags are filled first, and your original 3,000 hashes are kept.
- It downloads `meta.db` (3.5 GB of metadata, no binaries) to the instance.

### 3.3 Benign files

SOREL ships no benign binaries, so you supply them. Use several sources so the model can't learn "Windows system DLL vs everything else".

| Source | How | Expected | Command |
|---|---|---|---|
| Your existing ~6.4k Windows files | Upload from your laptop (benign only) | ~6k | `python data/01d_upload_local_benign.py --dir <folder> --source legacy` |
| Windows Server 2016/2019/2022/2025 | Start one Windows EC2 per version (on-demand t3.large, ~1 h each), run the PowerShell harvester, terminate | 15–30k after dedup | `data/01a_benign_windows_harvest.ps1 -Bucket $MOE_BUCKET` |
| winget (third-party software) | Every file verified against the vendor SHA-256 in the manifest | 5–10k | `git clone --depth 1 https://github.com/microsoft/winget-pkgs /mnt/winget && python data/01b_benign_winget.py --repo /mnt/winget` |
| Assemblage (NeurIPS'24) | ~91k MSVC builds of GitHub projects on Hugging Face, streamed | up to 25k | `python data/01c_benign_assemblage.py --inspect` then `python data/01c_benign_assemblage.py --max-files 25000` |

- The **target** is about 70k benign. More benign samples in the test set make the low-FPR numbers meaningful: TPR@FPR=1e-3 needs ≥10k benign test samples.
- **Groups** (one DLL name, one winget package, one GitHub repo) are capped at 20 files. Each group lands in exactly one split, and `build_dataset.py` asserts this.
- If you can get **VirusTotal Academic** or **ReversingLabs** access, SOREL's own benign set becomes available. That is the gold standard, so mention it if a reviewer asks.
- **BODMAS** gives malware binaries with family labels on request (institutional email, see whyisyoung.github.io/BODMAS). It is optional and would let you add a held-out-family test.

### 3.4 Merge and chunk

```bash
python data/02_build_manifest.py --malware manifest_malware.parquet
```

This prints rows by class, split and source. Check that every source appears in train, val and test.

---

## 4. Extract features (Stage A, then Stage B)

### 4.1 Pilot on the idle box (do not skip)

```bash
python extract/worker.py --max-chunks 1
python tools/progress.py --detail
```

Check four things:
- `r2` coverage should be > 0.8 on benign, and lower on packed malware is normal.
- `unk` (SAFE UNK rate) should be < 0.3. Close to 1.0 means the normalisation is wrong; stop and fix it.
- `n_edges > 0` on most samples. If not, the radare2 call-graph JSON changed format; fix `r2_extract.py` before scaling.
- `sec/sample` tells you how many CPU-hours remain: `progress.py` prints this.

### 4.2 Scale out

```bash
./infra/launch_spot.sh extract 3          # 3 × 32 vCPU
watch -n 300 python tools/progress.py     # from the idle box or your laptop
```

- **Sizing:** Stage A CPU-hours ≈ samples × sec/sample ÷ 3600. At ~20 s/sample, 120k samples is ~670 CPU-hours: about 7 hours on 3 × 32-vCPU instances.
- **Cost:** check the current spot price with `aws ec2 describe-spot-price-history --instance-types c6a.8xlarge --product-descriptions Linux/UNIX --max-items 5`, then multiply by instance-hours.
- **If instances are interrupted,** rerun the same `launch_spot.sh` line when progress stalls.

### 4.3 SAFE embeddings

```bash
./infra/launch_spot.sh safe 1             # reads token ids, writes v2/emb_safe/
```

### 4.4 Build the dataset

Run this on the idle box, which has plenty of RAM:

```bash
python build/build_dataset.py --graphs
aws s3 sync data_v2 s3://$MOE_BUCKET/v2/data_v2/
```

Copy `data_v2` to your laptop if you want to train there. It only holds features, never binaries.

`data_v2/coverage.md` is the expert-coverage table for the paper.

---

## 5. Train, evaluate, ablate, tabulate

```bash
python ablations/run_ablations.py --sweep-dropout        # pick expert dropout on VAL only
#   -> set train.expert_dropout in config.yaml
python ablations/run_ablations.py                        # 11 MoE variants × 3 seeds + 4 baselines
python report/make_tables.py                             # results/tables.md
```

| Piece | Where to run |
|---|---|
| MoE variants | Laptop CPU is fine (embeddings only) |
| LightGBM (2048 leaves, ~85k × 2568) | 32-vCPU box (≈ minutes); slow on a laptop |
| GIN | Idle box or a `g4dn.xlarge` spot |

**What the tables give you, mapped to the panel's comments:**

| Panel comment | Answer in `results/tables.md` |
|---|---|
| "Stronger comparison with state of the art" | Detection table: MalConv2-GCT (zero-shot), EMBER+LightGBM, SOREL-20M FFNN and a call-graph GIN, all on the same split, with TPR at FPR 1e-3/1e-2 |
| (same) | Behaviour-tag table: SOREL FFNN tag head and per-tag LightGBM vs MoE kNN voting |
| (same) | Retrieval table: every model's embedding under the same kNN protocol |
| (same) | Ablation table: dual vs single gate, concat, mean, gate input, expert dropout, node2vec vs LDP, single experts |
| "Dataset expansion" | 50k malware + ~70k benign from four benign sources, a temporal split (SOREL's own boundaries), group-disjoint benign, and the coverage table |
| Application framing (ESWA/EAAI) | Triage table: abstain on the most novel samples and route them to a sandbox, giving accuracy, recall and FPR on what is kept |

---

## 6. Graph expert: LDP or node2vec

`config.yaml → train.graph_expert: ldp` is the recommended default for the reason in §0. Report node2vec as the legacy-expert ablation.

If node2vec happens to score higher, still keep LDP as the main model and say why. A result that rests on per-binary node2vec embeddings is hard to defend in review.

---

## 7. Safety rules

- **Never download a SOREL object to your laptop.** All malware handling stays on the cloud instances.
- **Nothing is ever executed.** radare2 only disassembles, and SOREL binaries are additionally disarmed.
- **Instances have no inbound ports.** You reach them through Session Manager.
- **Keep benign binaries private** (Microsoft files and GitHub-licensed builds). Publish SHA-256 lists and the code. The data statement can say: "hashes and feature vectors available; binaries obtainable from the original sources".
- **Tear down at the end:** `aws ec2 describe-instances` should list nothing tagged `project=moe-paper`. Keep the bucket, since storage is cheap, but you can delete `v2/leases/`.

---

## 8. File map

| Path | Role |
|---|---|
| `config.yaml` | every knob (bucket, sizes, timeouts, training) |
| `common/pe_parity.py` | header parity (model view / analysis view) |
| `common/r2_extract.py` | one radare2 pass → SAFE tokens + call graph |
| `common/graph_feats.py` | ACFG node attributes, LDP, node2vec |
| `common/ember_feats.py` | thrember EMBER features |
| `common/s3io.py` | streaming reads, parquet writes, leases |
| `data/00…02` | manifests (malware sampling, benign collection, merge + chunk) |
| `extract/worker.py` | Stage A (spot-safe, resumable) |
| `embed/embed_safe.py` | Stage B |
| `build/build_dataset.py` | dataset.npz, splits, graphs, coverage |
| `model/moe.py`, `model/train_moe.py` | MoE + ablation variants |
| `eval/metrics.py`, `eval/evaluate.py` | all metrics (val-only selection) |
| `baselines/*.py` | MalConv2 zero-shot, EMBER+LightGBM, SOREL FFNN, call-graph GIN |
| `ablations/run_ablations.py` | dropout sweep + full grid, resumable |
| `report/make_tables.py` | `results/tables.md` |
| `infra/*` | bootstrap, spot launcher, IAM policy |
| `tests/smoke_test.py` | end-to-end check on synthetic data (no AWS) |
