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

1. **Generate Watermarked Text**  
   Create watermarked text at different temperature settings.

2. **Corrupt Watermarked Text**  
   Apply random edits or use round-trip translation to corrupt the watermarked text.

3. **Compute Pivotal Statistics**  
   Calculate the pivotal statistics for the edited text.

4. **Plot Power/Type II Errors**  
   Visualize the detection power (or Type II errors) across different scenarios:  
   - No edits  
   - Random edits  
   - Translation attacks  

5. **Compute Edit Tolerance Limits**  
   Determine the "edit tolerance limit."

6. **Adversarial Edits**  
   Perform adversarial edits on the watermarked text from Step 1, then compute pivotal statistics and plot the detection power.

