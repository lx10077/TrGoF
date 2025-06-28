# Codes of Tr-GoF for Robust Detection of Watermarks

This repository contains the code accompanying the paper:

> [Robust Detection of Watermarks for Large Language Models Under Human Edits](https://arxiv.org/abs/2411.13868)

If you find this repository useful in your research, please consider citing:

```bibtex
@article{li2024robust,
    title={Robust Detection of Watermarks for Large Language Models Under Human Edits},
    author={Li, Xiang and Ruan, Feng and Wang, Huiyuan and Long, Qi and Su, Weijie J},
    journal={arXiv preprint arXiv:2411.13868},
    year={2024}
}
```

---

## Key Idea

We introduce a **truncated family of goodness‑of‑fit (Tr‑GoF) tests** that remain powerful after human edits.

```python
import numpy as np

def compute_score(Ys, s: float = 2, eps: float = 1e-12,
                  mask: bool = True, first: int | None = None) -> float:
    '''Tr‑GoF test statistic.

    Parameters
    ----------
    Ys   : 2‑D np.ndarray, [n_sample, n_token]
        Pivotal probabilities in [0, 1].
    s    : float
        Divergence index. Supported: 2, 1, 0.5, 0, ‑1, and any ‑1 <= s <= 2.
    eps  : float
        Numerical stability constant.
    mask : bool
        Whether to truncate extreme order‑statistics for stability.
    first: int | None
        Use only the first `first` order‑statistics (truncation).

    Returns
    -------
    float
        a Tr-GoF score for each row, shape (n_sample, )
    '''
    # For Gumbel-max, p = 1 - Y since null = U(0, 1)
    ps = np.sort(1 - Ys)                     # compute p-value. 
    n  = ps.size
    rk = np.arange(1, (first or n) + 1) / n  # empirical CDF ranks
    if mask:                                 # perform truncation
        keep = (ps >= 1/n) & (rk >= ps)
        ps, rk = ps[keep], rk[keep]

    if s == 1:
        div = rk*np.log((rk+eps)/(ps+eps)) + (1-rk)*np.log((1-rk+eps)/(1-ps+eps))
    elif s == 0:
        div = ps*np.log((ps+eps)/(rk+eps)) + (1-ps)*np.log((1-ps+eps)/(1-rk+eps))
    elif s in (2, -1):
        base = rk if s == -1 else ps
        div  = (rk - ps)**2 / (base*(1-base) + eps) / 2
    elif s == 0.5:
        div = 2*((np.sqrt(rk)-np.sqrt(ps))**2 + (np.sqrt(1-rk)-np.sqrt(1-ps))**2)
    else:  # ‑1 < s < 0 or 0 < s < 1
        div = (1 - rk**s*(ps+eps)**(1-s) - (1-rk)**s*(1-ps+eps)**(1-s)) / (s*(1-s))
    return np.log(n * div.max()+eps)
```

Once all pivotal statistics are collected into an array (i.e., a matrix of shape `n_sample × n_token`), simply call `compute_score(Ys, s=2)` to obtain the Tr‑GoF statistic for each row.

For quick testing or integration, you may paste the function into your own project and estimate the critical value via simulation as follows:

```python
def compute_quantile(m, alpha, s, mask, eps=1e-12):
    """
    Estimate the (1 - alpha) quantile of the Tr-GoF test statistic under the null,
    based on repeated simulations from the uniform distribution.

    Parameters:
        m     : int     – Number of tokens (i.e., columns).
        alpha : float   – Significance level.
        s     : float   – Divergence parameter for the test.
        mask  : bool    – Whether to apply masking in compute_score.
        eps   : float   – Numerical stability constant.

    Returns:
        float – Estimated (1 - alpha) quantile of log-statistic.
    """
    qs = []
    for _ in range(10):
        raw_data = np.random.uniform(size=(10000, m))
        stats = compute_score(raw_data, s=s, mask=mask)
        q = np.quantile(np.log(stats + eps), 1 - alpha)
        qs.append(q)
    return np.mean(qs)
```

---

## Directory Structure

```plaintext
.
├── LLM_codes           # Code for language model experiments
├── simulation_codes    # Code for simulation studies
├── saved_fig_results   # Data and codes for generating plots
└── README.md
```

---

## Reproducing the Figures

To reproduce the plots in our paper, navigate to the `saved_fig_results` directory and follow the instructions provided in its README file.

---

## Pipeline for LLM Experiments

To reproduce the large language model experiments, first navigate to the `LLM_codes` directory:

```bash
cd LLM_codes
```

Then follow the steps below to run the experiments.


### 1. Generate Watermarked Text

This step creates watermarked text at various temperature settings. For example, the following command uses the OPT-1.3B model to generate 1,000 texts, each with a length of 400 tokens. It processes them in batches of 10, using the temperatures 0.1, 0.3, 0.5, and 0.7 in sequence:

```
python Step1_watermark_text.py \
  --model "facebook/opt-1.3b" \
  --c 5 \
  --m 400 \
  --T 1000 \
  --batch_size 10 \
  --all_temp 0.1 0.3 0.5 0.7
```

---

### 2. Corrupt Watermarked Text

This step applies various editing operations or round-trip translation to the previously generated watermarked text from Step 1. By default, Step 1 saves all generated texts, and now you can corrupt them using the following editing methods:

- **Substitution**: Replace characters or tokens with random ones.
- **Deletion**: Randomly remove parts of the text.
- **Insertion**: Insert new random characters or tokens.
- **Round-trip Translation**: Translate the text to another language and back.

Including the `--substitution` flag, for instance, enables the substitution corruption method. Similarly, `--deletion`, `--insertion`, and `--translation` flags enable their respective methods. 

For example:

```
python Step2_corrupt_text.py \
  --model "facebook/opt-1.3b" \
  --c 5 \
  --m 400 \
  --T 1000 \
  --batch_size 10 \
  --all_temp 0.1 0.3 0.5 0.7 \
  --substitution \
  --deletion \
  --insertion \
  --translation
```

---

### 3. Compute Pivotal Statistics

This step calculates pivotal statistics for the edited (or corrupted) texts generated in Step 2. It processes all temperatures and editing methods used previously, so the command remains largely unchanged:

```bash
python Step3_compute_Y.py \
  --model "facebook/opt-1.3b" \
  --c 5 \
  --m 400 \
  --T 1000 \
  --all_temp 0.1 0.3 0.5 0.7 \
  --substitution \
  --deletion \
  --insertion \
  --translation
```

---

### 4. Plot Type II Errors

In this step, we visualize the detection power (or Type II errors) under various scenarios.

- **No Edits:**  Evaluate Type II errors when all text is watermarked, without any modifications.

  ``` 
  python Step4_plot_power.py \
    --model "facebook/opt-1.3b" \
    --c 5 \
    --m 400 \
    --T 1000 \
    --all_temp 0.1 0.3 0.5 0.7 \
    --alpha 0.01 
  ```

- **Random Edits:** Assess performance under random edits (substitution, deletion, and insertion).

  ```
  python Step4_plot_robust.py \
    --model "facebook/opt-1.3b" \
    --c 5 \
    --m 400 \
    --T 1000 \
    --all_temp 0.1 0.3 0.5 0.7 \
    --alpha 0.01\
    --substitution \
    --deletion \
    --insertion
  ```

- **Translation Edits:** Check results after applying round-trip translation.

  ```
  python Step4_plot_trans.py \
    --model "facebook/opt-1.3b" \
    --c 5 \
    --m 400 \
    --T 1000 \
    --all_temp 0.1 0.3 0.5 0.7 \
    --alpha 0.01
  ```
 
---

### 5. Compute Edit Tolerance Limits

Determine the “edit tolerance limit” by sequentially applying three types of random edits. For example, to estimate the edit tolerance limit using the OPT-1.3B model and Sheared-LLaMA-2.7B model at a temperature of 1 and a significance level \(\alpha = 0.01\), run:

```
python Task1_poem.py \
  --c 5 \
  --m 400 \
  --T 1000 \
  --temp 1 \
  --alpha 0.01
```

---

### 6. Adversarial Edits

This step performs adversarial edits on the watermarked text generated in Step 1, then computes pivotal statistics and plots detection power under these adversarial conditions.

```
python Task2_adversarial_edit.py \
  --model "facebook/opt-1.3b" \
  --c 5 \
  --m 400 \
  --T 1000 \
  --all_temp 0.1 0.3 0.5 0.7 \
  --alpha 0.01
```

