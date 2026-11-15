# MantisV2 reproduction notes

Paper: Feofanov et al. (2026), *MantisV2: Closing the Zero-Shot Gap in Time Series Classification with
Synthetic Data and Test-Time Strategies*, arXiv:2602.17868 — Tables 2 (HAR) and 3 (EEG), MantisV2 column.

The group reproduced the frozen MantisV2 and Random Forest baseline on 17 of the 18 datasets. See [dataset setup](DATASETS.md) for loading instructions and [combined results](../results/RESULTS.md) for the full comparisons.

## Protocol (from paper + official repo README)

- Code: `pip install mantis-tsfm` (github.com/vfeofanov/mantis), checkpoint `paris-noah/MantisV2` (4.19M params, paper says 4.2M).
- `MantisV2(return_transf_layer=2, output_token='combined')` → 512-dim embedding per channel (cls token ‖ mean token).
  The repo tutorial re-sets both attributes after `from_pretrained`; we did the same.
- Every series linearly interpolated to length 512; each channel encoded independently, embeddings concatenated.
- Random Forest, 200 trees, unlimited depth, `random_state` = 0, 1, 2 → mean ± std (paper: "3 seeds").

## Datasets 1–6 — Aadil Shaji

The Random Forest results for datasets 2–6 come from [run 05](../kaggle_runs/05_part1_seven_datasets/). [Run 09](../kaggle_runs/09_part5_reruns/)
contains the LoRA reruns for EMOPain and MP50. Ego4D requires a signed licence and was skipped.

| # | Dataset | Source and split used | Train / test | Ours (mean ± std) | Paper | RF seeds (0 / 1 / 2) |
|---|---|---|---|---|---|---|
| 1 | Ego4D | Ego4D IMU data, signed licence required | — | Not run | 0.5258 | — |
| 2 | [EMOPain](../results/by_dataset/02_EMOPain.csv) | UEA `EMOPain.zip`, archive split | 968 / 355 | 0.8826 ± 0.0058 | 0.8798 | 0.8901 / 0.8817 / 0.8761 |
| 3 | [HHAR-ID](../results/by_dataset/03_HHAR-ID.csv) | WOODS `HHAR.h5`, domains merged and split with seed 0 | 8,717 / 3,418 | 0.9834 ± 0.0005 | 0.9845 | 0.9839 / 0.9827 / 0.9836 |
| 4 | [HHAR-OOD](../results/by_dataset/04_HHAR-OOD.csv) | WOODS `HHAR.h5`, Nexus4/S3/S3-mini train, LG Watch test | 11,150 / 2,154 | 0.5868 ± 0.0038 | 0.5822 | 0.5915 / 0.5868 / 0.5822 |
| 5 | [MP8](../results/by_dataset/05_MP8.csv) | FM4MTSC `MP8.npy`, released train/test split | 1,426 / 595 | 0.6756 ± 0.0063 | 0.6857 | 0.6689 / 0.6840 / 0.6739 |
| 6 | [MP50](../results/by_dataset/06_MP50.csv) | FM4MTSC `MP50.npy`, released train/test split | 1,426 / 595 | 0.7092 ± 0.0049 | 0.7345 | 0.7042 / 0.7076 / 0.7160 |

HHAR-ID differs from the paper by one sample in each split count (paper 8,716 train / 3,419 test).
MP8 and MP50 use the released splits and match the paper's split sizes. Their reproduction is
roughly 1.0 and 2.5 accuracy points lower, respectively, so the data version or preprocessing is a possible cause.

## Datasets 7–12 — Padmanabh Khunt

Run on 6 Nov 2026, Google Colab (Tesla T4 GPU, Python 3.13), notebook
`MantisV2_Reproduction_Datasets_7-12_Padmanabh.ipynb` in Padmanabh's Google Drive
(https://colab.research.google.com/drive/16mi0PGWg-hHSUL-BoEAVT5veu6RLIiDv).

**Update (same day):** the Colab compute quota ran out while CAP was running, so the full run moved to Kaggle:
private notebook *MantisV2 Reproduction Datasets 7-12 Padmanabh*
(https://www.kaggle.com/code/padmanabhkhunt/mantisv2-reproduction-datasets-7-12-padmanabh), CPU session
(4 cores, 31 GB RAM), run as "Save & Run All (Commit)" version 1. In an interactive test on Kaggle, FingerMovements
(0.5500 ± 0.0082), Blink (0.9956 ± 0.0018) and UCI-HAR seed 0 (0.9033) gave **identical** numbers to Colab —
the pipeline is deterministic across machines. CAP embeddings are saved as `cap_embeddings_Machine*.npz`
in the Kaggle output for the improvement experiments. The same pipeline as a
script is `src/reproduce_mantisv2.py`.

### Data sources

| # | Dataset | Source used | Shape check vs paper Tables 5/6 |
|---|---|---|---|
| 7 | UCI-HAR | TF-C preprocessed (figshare article 19930244, `train.pt`/`test.pt`, via figshare API) | train (5881, 3, 206), test 2947 — matches |
| 8 | Blink | timeseriesclassification.com `Blink.zip` (same as UniTS script) | train (500, 4, 510), test 450 — matches |
| 9 | CAP-ID | WOODS preprocessed `CAP.h5` (Google Drive id 1NFwX2CqLrenWF4az0c6J-OglAoD48PAT, 8.66 GB zip / 9.76 GB h5) | 40,390 samples total; train 25,748 / test 10,098 — matches |
| 10 | CAP-OOD | same file: Machines 0,1,3 train / Machine 2 val (unused) / Machine 4 test | train 27,393, test 8,265 — matches |
| 11 | Epilepsy-EEG | TF-C preprocessed (figshare article 19930199) | train (60, 1, 178), test 11,420 — matches |
| 12 | FingerMovements | timeseriesclassification.com `FingerMovements.zip` (UEA) | train (316, 28, 50), test 100 — matches |

### Results (accuracy, 3 RF seeds) — final Kaggle run (version 1, 7,186 s, CPU only)

| Dataset | Ours | Paper (MantisV2) | Diff | Per seed (0 / 1 / 2) |
|---|---|---|---|---|
| UCI-HAR | 0.9013 ± 0.0015 | 0.9013 ± 0.002 | 0.0000 | 0.9033 / 0.8999 / 0.9006 |
| Blink | 0.9956 ± 0.0018 | 0.9956 ± 0.002 | 0.0000 | 0.9978 / 0.9956 / 0.9933 |
| CAP-ID | 0.8091 ± 0.0010 | 0.8152 ± 0.001 | **−0.0061** | 0.8106 / 0.8086 / 0.8082 |
| CAP-OOD | 0.7858 ± 0.0018 | 0.7859 ± 0.000 | −0.0001 | 0.7851 / 0.7840 / 0.7883 |
| Epilepsy-EEG | 0.9558 ± 0.0012 | 0.9558 ± 0.001 | 0.0000 | 0.9542 / 0.9561 / 0.9570 |
| FingerMovements | 0.5500 ± 0.0082 | 0.5500 ± 0.010 | 0.0000 | 0.54 / 0.56 / 0.55 |

5 of 6 match the paper to 3–4 decimals. CAP-ID is 0.6 points lower — the only dataset whose split depends on an
unreported random seed (issue 3 below), so a different train/test draw is the most likely cause; CAP-OOD uses the same
embeddings with a fixed machine split and matches, which suggests the embeddings themselves are correct.
(The earlier Colab run gave UCI-HAR 0.9014 ± 0.0014; Kaggle 0.9013 — tiny CPU/GPU float difference.)

Timing on Kaggle CPU (torch used only 2 threads): CAP embedding 3,386 s for 40,390 samples; RF 1,596 s (CAP-ID) and
1,681 s (CAP-OOD) for 3 seeds.

The executed notebook (`.ipynb`, `.html`, `.pdf`) and `mantisv2_results_datasets_7_12.csv` are in
[`kaggle_runs/01_reproduction_datasets_7-12/`](../kaggle_runs/01_reproduction_datasets_7-12/).

### Reproduction issues found

1. **Dependency clash:** `aeon` (data loader) cannot be imported on current Colab — numba requires NumPy ≤ 2.3, Colab ships NumPy 2.4. Worked around by parsing the `.ts` files directly (no change to data).
2. **figshare blocks scripted downloads:** the TF-C `ndownloader/articles/...` zip links return HTTP 202 with an empty body (bot check) to `wget`. The official figshare REST API (`api.figshare.com/v2/articles/<id>`) gives direct per-file links that work.
3. **Unreported split randomisation for CAP-ID:** paper says "merge all domains, 63.75% / 11.25% / 25% split" but gives no seed or stratification. We used a plain random permutation (seed 0). Sizes match; exact samples may not.
4. **Paper's dataset table vs. TF-C files:** shapes for UCI-HAR and Epilepsy match the paper exactly (3 channels × 206 for UCI-HAR), so the paper's "preprocessed following Zhang et al. (2022)" means the TF-C figshare release.
5. **Compute:** CAP is 19 channels × 3000 steps. Embedding all 40k samples took ~2 min on a T4, but the Random Forest on 25k × 9,728 features is slow on free Colab (2 CPU cores); the Colab UI showed "Connecting" while both cores were busy.
6. **Default classifier variance:** with only 100 test samples, FingerMovements swings ±1 sample per seed (0.54–0.56), so the paper's 0.55 ± 0.01 is close to chance level for this binary task.

## Datasets 13–18 — Pranet Pednekar

The PCL-ID Random Forest result was saved in [run 06](../kaggle_runs/06_part2_PCL_failed_out_of_RAM/) before the token-pooling stage ran out of RAM.
[Run 09](../kaggle_runs/09_part5_reruns/) includes the completed PCL-OOD run and the remaining PCL-ID methods.
SEDFx-ID and SEDFx-OOD ran separately in [run 07](../kaggle_runs/07_part3_SEDFx-ID/) and [run 08](../kaggle_runs/08_part4_SEDFx-OOD/). The two SCP datasets are in the shared [run 05](../kaggle_runs/05_part1_seven_datasets/).

| # | Dataset | Source and split used | Train / test | Ours (mean ± std) | Paper | RF seeds (0 / 1 / 2) |
|---|---|---|---|---|---|---|
| 13 | [PCL-ID](../results/by_dataset/13_PCL-ID.csv) | WOODS `PCL.h5`, domains merged and split with seed 0 | 14,406 / 5,650 | 0.5769 ± 0.0031 | 0.5716 | 0.5726 / 0.5798 / 0.5782 |
| 14 | [PCL-OOD](../results/by_dataset/14_PCL-OOD.csv) | WOODS `PCL.h5`, `Cho2017` train, `Lee2019_MI` test | 9,880 / 7,800 | 0.5343 ± 0.0024 | 0.5311 | 0.5322 / 0.5377 / 0.5329 |
| 15 | [SEDFx-ID](../results/by_dataset/15_SEDFx-ID.csv) | WOODS `SEDFx.h5`, age domains merged and split with seed 0 | 152,178 / 59,678 | 0.7985 ± 0.0004 | 0.8000 | 0.7980 / 0.7984 / 0.7991 |
| 16 | [SEDFx-OOD](../results/by_dataset/16_SEDFx-OOD.csv) | WOODS `SEDFx.h5`, ages 20–60 train, ages 80–100 test | 133,746 / 52,838 | 0.7640 ± 0.0002 | 0.7636 | 0.7640 / 0.7642 / 0.7638 |
| 17 | [SelfRegulationSCP1](../results/by_dataset/17_SelfRegulationSCP1.csv) | UEA `SelfRegulationSCP1`, archive split | 268 / 293 | 0.8134 ± 0.0016 | 0.8134 | 0.8157 / 0.8123 / 0.8123 |
| 18 | [SelfRegulationSCP2](../results/by_dataset/18_SelfRegulationSCP2.csv) | UEA `SelfRegulationSCP2`, archive split | 200 / 180 | 0.5148 ± 0.0228 | 0.5167 | 0.5111 / 0.4889 / 0.5444 |

PCL-ID has one more training sample than the paper (14,406 instead of 14,405).
For the PCL and SEDFx OOD runs, the `PhysionetMI` and age 60–80 validation domains were left unused.
