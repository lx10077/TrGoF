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

## Directory Structure

```plaintext
.
├── LLM_codes           # Code for language model experiments
├── simulation_codes    # Code for simulation studies
├── saved_fig_results   # Data used for generating plots
└── README.md
```

## Pipeline for LLM Experiments

Follow these steps to reproduce the results:

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

