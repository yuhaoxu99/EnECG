# EnECG
Official code for "EnECG: Efficient Ensemble Learning for Electrocardiogram Multi-task Foundation Model".

## Prepare Dataset
Prepare ECG Data from [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/) and download our prepared [Subset Data and Label](https://drive.google.com/drive/folders/1IkHkwa0HUbxmieBHMPd-VRdYQJbKLm3P?usp=share_link).

We provide `.jsonl` file subset from the MIMIC-IV-ECG, along with the corresponding labels to evaluate in different downstream tasks, including RR Interval Estimation `rr_interval`, Age Estimation `age`, Gender Classification `gender`, Potassium Abnormality Prediction `flag`, and Arrhythmia Detection `report_label`.

## Prepare checkpoints
Download TEMPO and ECG-FM through [Checkpoints](https://drive.google.com/drive/folders/19yAkDf2yFHaWQ0cDp3McuMsFtl1c4aSY?usp=share_link).

## Installation
The required packages can be installed by running `pip install -r requirements.txt`.

For `ECG-FM` environment please refer the link [ECG-FM](https://github.com/bowang-lab/ECG-FM) and [fairseq-signals](https://github.com/Jwoo5/fairseq-signals).
