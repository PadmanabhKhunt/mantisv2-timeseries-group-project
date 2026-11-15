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

Results below use test accuracy. Difference is the reproduced mean minus the paper's reported accuracy.

## Datasets 1–6 — Aadil Shaji

### Run records

[Run 05](../kaggle_runs/05_part1_seven_datasets/) contains the five completed baseline runs. [Run 09](../kaggle_runs/09_part5_reruns/) contains the EMOPain and MP50 LoRA reruns.

### Data and results

| # | Dataset | Source and split used | Train / test | Ours (mean ± std) | Paper | Difference | RF seeds (0 / 1 / 2) |
|---|---|---|---|---|---|---|---|
| 1 | Ego4D | Ego4D IMU data, signed licence required | — | Not run | 0.5258 | — | — |
| 2 | [EMOPain](../results/by_dataset/02_EMOPain.csv) | UEA `EMOPain.zip`, archive split | 968 / 355 | 0.8826 ± 0.0058 | 0.8798 | +0.0028 | 0.8901 / 0.8817 / 0.8761 |
| 3 | [HHAR-ID](../results/by_dataset/03_HHAR-ID.csv) | WOODS `HHAR.h5`, domains merged and split with seed 0 | 8,717 / 3,418 | 0.9834 ± 0.0005 | 0.9845 | -0.0011 | 0.9839 / 0.9827 / 0.9836 |
| 4 | [HHAR-OOD](../results/by_dataset/04_HHAR-OOD.csv) | WOODS `HHAR.h5`, Nexus4/S3/S3-mini train, LG Watch test | 11,150 / 2,154 | 0.5868 ± 0.0038 | 0.5822 | +0.0046 | 0.5915 / 0.5868 / 0.5822 |
| 5 | [MP8](../results/by_dataset/05_MP8.csv) | FM4MTSC `MP8.npy`, released train/test split | 1,426 / 595 | 0.6756 ± 0.0063 | 0.6857 | -0.0101 | 0.6689 / 0.6840 / 0.6739 |
| 6 | [MP50](../results/by_dataset/06_MP50.csv) | FM4MTSC `MP50.npy`, released train/test split | 1,426 / 595 | 0.7092 ± 0.0049 | 0.7345 | -0.0253 | 0.7042 / 0.7076 / 0.7160 |

### Reproduction notes

- Ego4D required a signed licence. HHAR-ID split counts differ from the paper by one sample in each split.
- MP8 and MP50 use the released splits. Their scores are about 1.0 and 2.5 percentage points below the paper, respectively. Data version or preprocessing differences may contribute.

## Datasets 7–12 — Padmanabh Khunt

### Run records

[Run 01](../kaggle_runs/01_reproduction_datasets_7-12/) contains all six baseline runs. The final run used a Kaggle CPU session after the Colab quota ran out during CAP.

### Data and results

| # | Dataset | Source and split used | Train / test | Ours (mean ± std) | Paper | Difference | RF seeds (0 / 1 / 2) |
|---|---|---|---|---|---|---|---|
| 7 | [UCI-HAR](../results/by_dataset/07_UCI-HAR.csv) | TF-C figshare 19930244, released train/test files | 5,881 / 2,947 | 0.9013 ± 0.0015 | 0.9013 | 0.0000 | 0.9033 / 0.8999 / 0.9006 |
| 8 | [Blink](../results/by_dataset/08_Blink.csv) | UEA `Blink.zip`, archive split | 500 / 450 | 0.9956 ± 0.0018 | 0.9956 | 0.0000 | 0.9978 / 0.9956 / 0.9933 |
| 9 | [CAP-ID](../results/by_dataset/09_CAP-ID.csv) | WOODS `CAP.h5`, machines merged and split with seed 0 | 25,748 / 10,098 | 0.8091 ± 0.0010 | 0.8152 | -0.0061 | 0.8106 / 0.8086 / 0.8082 |
| 10 | [CAP-OOD](../results/by_dataset/10_CAP-OOD.csv) | WOODS `CAP.h5`, machines 0, 1 and 3 train, machine 4 test | 27,393 / 8,265 | 0.7858 ± 0.0018 | 0.7859 | -0.0001 | 0.7851 / 0.7840 / 0.7883 |
| 11 | [Epilepsy-EEG](../results/by_dataset/11_Epilepsy-EEG.csv) | TF-C figshare 19930199, released train/test files | 60 / 11,420 | 0.9558 ± 0.0012 | 0.9558 | 0.0000 | 0.9542 / 0.9561 / 0.9570 |
| 12 | [FingerMovements](../results/by_dataset/12_FingerMovements.csv) | UEA `FingerMovements.zip`, archive split | 316 / 100 | 0.5500 ± 0.0082 | 0.5500 | 0.0000 | 0.5400 / 0.5600 / 0.5500 |

### Reproduction notes

- CAP-ID was about 0.6 percentage points below the paper. The paper gives no seed for the ID split. CAP-OOD used a fixed split with machine 2 held out for validation and unused in this run.
- FingerMovements has 100 test samples, so one changed prediction moves accuracy by one percentage point.

## Datasets 13–18 — Pranet Pednekar

### Run records

PCL-ID's Random Forest result was saved in [run 06](../kaggle_runs/06_part2_PCL_failed_out_of_RAM/) before the token-pooling stage ran out of RAM. [Run 09](../kaggle_runs/09_part5_reruns/) completed the PCL experiments. The SEDFx runs are in [run 07](../kaggle_runs/07_part3_SEDFx-ID/) and [run 08](../kaggle_runs/08_part4_SEDFx-OOD/). The two SCP datasets are in [run 05](../kaggle_runs/05_part1_seven_datasets/).

### Data and results

| # | Dataset | Source and split used | Train / test | Ours (mean ± std) | Paper | Difference | RF seeds (0 / 1 / 2) |
|---|---|---|---|---|---|---|---|
| 13 | [PCL-ID](../results/by_dataset/13_PCL-ID.csv) | WOODS `PCL.h5`, domains merged and split with seed 0 | 14,406 / 5,650 | 0.5769 ± 0.0031 | 0.5716 | +0.0053 | 0.5726 / 0.5798 / 0.5782 |
| 14 | [PCL-OOD](../results/by_dataset/14_PCL-OOD.csv) | WOODS `PCL.h5`, `Cho2017` train, `Lee2019_MI` test | 9,880 / 7,800 | 0.5343 ± 0.0024 | 0.5311 | +0.0032 | 0.5322 / 0.5377 / 0.5329 |
| 15 | [SEDFx-ID](../results/by_dataset/15_SEDFx-ID.csv) | WOODS `SEDFx.h5`, age domains merged and split with seed 0 | 152,178 / 59,678 | 0.7985 ± 0.0004 | 0.8000 | -0.0015 | 0.7980 / 0.7984 / 0.7991 |
| 16 | [SEDFx-OOD](../results/by_dataset/16_SEDFx-OOD.csv) | WOODS `SEDFx.h5`, ages 20–60 train, ages 80–100 test | 133,746 / 52,838 | 0.7640 ± 0.0002 | 0.7636 | +0.0004 | 0.7640 / 0.7642 / 0.7638 |
| 17 | [SelfRegulationSCP1](../results/by_dataset/17_SelfRegulationSCP1.csv) | UEA `SelfRegulationSCP1`, archive split | 268 / 293 | 0.8134 ± 0.0016 | 0.8134 | 0.0000 | 0.8157 / 0.8123 / 0.8123 |
| 18 | [SelfRegulationSCP2](../results/by_dataset/18_SelfRegulationSCP2.csv) | UEA `SelfRegulationSCP2`, archive split | 200 / 180 | 0.5148 ± 0.0228 | 0.5167 | -0.0019 | 0.5111 / 0.4889 / 0.5444 |

### Reproduction notes

- PCL-ID has one more training sample than the paper (14,406 instead of 14,405).
- The OOD runs left out the validation domains: `PhysionetMI` for PCL and ages 60–80 for SEDFx.
