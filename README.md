# EnECG
Official code for "EnECG: Efficient Ensemble Learning for Electrocardiogram Multi-task Foundation Model".

## Framework of EnECG

<div align="center">
  <img src="https://raw.githubusercontent.com/yuhaoxu99/EnECG/main/img/EnECG.png" alt="architecture" width="500"/>
  <p><i>Figure 1: The framework of EnECG. The EnECG framework comprises three main steps.<br><br>
  (1) Each pretrained foundation model (M₁, M₂, ..., Mₙ) requires a specific input length, so we downsample the ECG signals and feed them into the frozen models. A feed-forward network (FFN) is added and fine-tuned to obtain (M<sub>ϕ1</sub>, M<sub>ϕ2</sub>, ..., M<sub>ϕₙ</sub>).<br><br>
  (2) To reduce training costs, a subset of ECG leads is selected and fed into a Mixture of Experts (MoE), which outputs gating probabilities W.<br><br>
  (3) Finally, the results are ensembled via a weighted sum: ŷ = ∑<sub>i=1</sub><sup>N</sup> W<sub>i</sub> ŷ<sub>i</sub>.
  </i></p>
</div>

## Prepare Dataset
Prepare ECG Data from [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/) and download our prepared [Subset Data and Label](https://drive.google.com/drive/folders/1IkHkwa0HUbxmieBHMPd-VRdYQJbKLm3P?usp=share_link).

We provide `.jsonl` file subset from the MIMIC-IV-ECG, along with the corresponding labels to evaluate in different downstream tasks, including RR Interval Estimation `rr_interval`, Age Estimation `age`, Gender Classification `gender`, Potassium Abnormality Prediction `flag`, and Arrhythmia Detection `report_label`.

## Prepare checkpoints
Download TEMPO and ECG-FM through [Checkpoints](https://drive.google.com/drive/folders/19yAkDf2yFHaWQ0cDp3McuMsFtl1c4aSY?usp=share_link).

## Installation
The required packages can be installed by running `pip install -r requirements.txt`.

For `ECG-FM` environment please refer the link [ECG-FM](https://github.com/bowang-lab/ECG-FM) and [fairseq-signals](https://github.com/Jwoo5/fairseq-signals).

## 🚀Quick Start
In the `run.sh`, we provide shell scripts, and you can change the `--label`, `--task_name` and `--num_class` to start running.
