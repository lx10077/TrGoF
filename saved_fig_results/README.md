# Figure and Table Reproduction for Tr-GoF: Robust Detection of Watermarks

This repository contains the code for reproducing the figures and tables in the paper:

> [**Robust Detection of Watermarks for Large Language Models Under Human Edits**](https://arxiv.org/abs/2411.13868)

If you find this repository helpful for your research, please consider citing:

```bibtex
@article{li2024robust,
    title={Robust Detection of Watermarks for Large Language Models Under Human Edits},
    author={Li, Xiang and Ruan, Feng and Wang, Huiyuan and Long, Qi and Su, Weijie J},
    journal={arXiv preprint arXiv:2411.13868},
    year={2024}
}
```

## How to Reproduce Figures and Tables

To run the scripts below, make sure you are in the `saved_fig_results` directory. You can do this by running:

```bash
cd TrGoF/saved_fig_results
```

---

Of course! Here's the updated and polished version of your README section, with consistent formatting, corrected typos, and improved clarity:

---

## How to Reproduce Figures and Tables

To run the scripts below, make sure you are in the `saved_fig_results` directory. You can do this by running:

```bash
cd TrGoF/saved_fig_results
```

---

### 📖 Introduction Figures

- **Figure 2, 3** — *Overview/Introduction Plot*  
  ```bash
  python 0intro/plot_intro.py  # This script generates both figures
  ```

---

### 📗 Simulation Figures

- **Figures 5, 13, 14** — *Histograms*  
  ```bash
  python 7histogram/plot_hist_TrGoF.py   # For Figures 5 and 13  
  python 7histogram/plot_hist_HC.py      # For Figure 14
  ```

- **Figures 6, 15** — *Empirical Detection Boundaries*  
  ```bash
  python 8contour/plot_TrGoF_boundary.py   # For Figure 6  
  python 8contour/plot_HC_boundary.py      # For Figure 15
  ```

- **Figure 7** — *Failure of Existing Detection Rules*  
  ```bash
  python 8contour/plot_sumrule_boundary.py
  ```

---

### 📘 LLM Experiment Figures

- **Figures 8, 9, 16** — *Statistical Power*  
  ```bash
  python 1power/plot_power_from_data.py
  ```

- **Figures 10, 17** — *Robustness Evaluation*  
  ```bash
  python 2robust/plot_robust_from_data.py
  ```

- **Figures 12, 18** — *Round-Trip Translation*  
  ```bash
  python 3translation/plot_trans_from_data.py
  ```

- **Tables 1, 2** — *Edit Tolerance Limits*  
  ```bash
  python 4poem/compute_edit_tolerance_limit.py
  ```

- **Figure 4** — *$P_{\Delta}$-Efficiency*  
  ```bash
  python 5efficiency/plot_efficiency.py
  ```

- **Figures 11, 19, 20** — *Adversarial Edits*  
  ```bash
  python 6adversarial/plot_adv_from_data.py         # For Figure 11
  python 6adversarial/plot_adv_from_data_all.py     # For Figures 19 and 20
  ```
