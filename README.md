<h1 align="center">Correction Training for Diffusion Models</h1>

<p align="center">
  <a href="https://sunovivid.github.io/">Donghoon Ahn</a><sup>1</sup> &nbsp;
  <a href="https://raywang4.github.io/">Runqian Wang</a><sup>1</sup> &nbsp;
  <a href="https://people.eecs.berkeley.edu/~efros/">Alexei A. Efros</a><sup>1,*</sup> &nbsp;
  <a href="https://taesung.me/">Taesung Park</a><sup>2,*</sup> &nbsp;
  <a href="https://yashbelhe.github.io/">Yash Belhe</a><sup>2,*</sup>
  <br>
  <sup>1</sup> UC Berkeley &emsp; <sup>2</sup> Reve
  <br>
  <sub>* Corresponding authors</sub>
</p>

<p align="center">
  <a href="https://sunovivid.github.io/correction-training/paper.pdf"><img src="assets/badges/paper.svg" alt="Paper: PDF"></a>
  <a href="https://sunovivid.github.io/correction-training/"><img src="assets/badges/project.svg" alt="Project page"></a>
  <a href="#code-and-checkpoints"><img src="assets/badges/code.svg" alt="Code: Coming soon"></a>
  <a href="#code-and-checkpoints"><img src="assets/badges/checkpoints.svg" alt="Checkpoints: Coming soon"></a>
</p>

<p align="center"><b>No guidance needed.</b></p>

<p align="center">
  Correction achieves <b>2.07 FID with DiT, 1.92 with SiT, and 2.96 with JiT</b><br>
  on class-conditional ImageNet-256, <b>without classifier-free guidance</b>.
</p>

<p align="center">
  <img src="assets/samples.png" width="100%" alt="ImageNet samples generated with correction: class-conditional generation without CFG, and unconditional generation.">
</p>

Diffusion models are trained on noisy real images, but sample from states produced by their own imperfect predictions. **Correction training** adds a network that learns to refine the denoiser's prediction using the same clean target. At inference, we reuse this corrector within each denoising step before advancing the sampling trajectory.

## Method

<p align="center">
  <img src="assets/method.png" width="100%" alt="Training and sampling diagram. Both networks learn from the same clean target, with stop-gradient through the denoiser prediction. Sampling contains an outer denoising loop and an inner correction loop.">
</p>

- **Training:** the denoiser predicts from a noisy image; the corrector takes that prediction and learns to recover the same clean target. Correction-loss gradients are stopped at the denoiser's prediction.
- **Sampling:** denoise once, apply the corrector repeatedly at the same noise level, then use the refined prediction to construct the next noisy state.

## ImageNet results

**High-quality generation without CFG.** Correction improves FID across DiT, SiT, SiT-REPA, and JiT at approximately the same sampling budget.

| Model | Base FID ↓ | Correction, no CFG ↓ | CFG reference ↓ |
| :--- | ---: | ---: | ---: |
| DiT-XL/2 | 9.58 | **2.07** | 2.51 |
| SiT-XL/2 | 8.55 | **1.92** | 2.11 |
| SiT-REPA-XL/2 | 5.53 | **1.31** | 1.35 |
| JiT-L/16 | 14.86 | **2.96** | 2.35 |

*Class-conditional ImageNet-256, FID on 50K samples, approximately 250 network evaluations per image. The budget includes denoiser, corrector, and CFG calls. See the paper for each model's settings.*

### Control refinement with K

Vary the number of corrections per denoising step to change the final sample. The examples below are **complete generations from noise**, with the same initial noise across each row.

<p align="center">
  <img src="assets/imagenet-k-sweep.png" width="100%" alt="Final SiT-REPA generations of a macaw and hourglass at K = 0, 1, 2, 4, and 8; DiT FID and precision–recall curves across correction counts.">
</p>

*Left: SiT-REPA final samples at different K. Right: DiT FID and precision–recall, evaluated on 50K images per setting. Denoising steps are adjusted to keep the total budget approximately fixed; more correction is not always better.*

<details>
<summary>More final-sample K sweeps: SiT, SiT-REPA, and JiT</summary>

<p align="center">
  <img src="assets/imagenet-more-k-sweeps.png" width="100%" alt="Final SiT, SiT-REPA, and JiT generations at K = 0, 1, 3, and 6, including bridges, cars, stockings, and people.">
</p>

*Each strip shares its class, initial noise, and denoising grid.*

</details>

### More diversity, less oversaturation than strong CFG

Increasing CFG can push samples toward similar close-up portraits and oversaturated colors. Correction preserves more varied poses, framings, and backgrounds while producing sharp, coherent images.

<p align="center">
  <img src="assets/imagenet-diversity.png" width="100%" alt="Same-class husky samples: correction retains varied poses, scenes, and colors, while strong CFG produces similar close-up faces and oversaturation. FID and recall are shown for every setting.">
</p>

*Same class and initial noises. Top: increasing correction K without CFG. Bottom: increasing CFG strength w. FID and recall quantify the quality–diversity trade-off.*

## Text-to-image results

### Already using CFG? Add correction.

**Correction also works alongside CFG.** It can reduce the “AI slop” look—oversaturated colors, overly stylized appearances, and repetitive compositions—while keeping more variety across seeds.

<p align="center">
  <img src="assets/cfg-plus-correction.png" width="100%" alt="Text-to-image results for 'a photo of a bench'. Four groups compare no CFG versus CFG, and no correction versus correction. CFG plus correction produces more varied bench designs and viewpoints.">
</p>

*Prompt: “a photo of a bench.” Each group contains the same eight seeds. Left: no CFG (w = 1); right: CFG (w = 6). Top: no correction; bottom: K = 6.*

In the paper's human study, **CFG + correction was preferred in 60.75% of quality votes and 60.25% of diversity votes** across 80 prompts. The comparison uses the released baseline with CFG (w = 6) and the correction-trained model with CFG + correction (w = 4, K = 8), with 800 votes per criterion.

### Better prompt alignment, with or without CFG

| Setting | K | Network evaluations | GenEval ↑ | DPG-Bench ↑ |
| :--- | ---: | ---: | ---: | ---: |
| No CFG | 0 | 250 | 28.30 | 70.71 |
| No CFG + correction | 6 | 245 | **50.09** | **76.82** |
| CFG (w = 4) | 0 | 250 | 68.08 | 83.38 |
| CFG + correction (w = 4) | 2 | 246 | **70.48** | **83.69** |

*NanoGen text-to-image model at 256 × 256. The sampling budget is approximately matched: 250 vs. 245 network evaluations without CFG, and 250 vs. 246 with CFG. Guided sampling applies correction before CFG.*

<p align="center">
  <img src="assets/text-to-image.png" width="100%" alt="Text-to-image comparisons of a green guitar, vintage car, sunlit kitchen, three violins, rabbit and seashell, and Victorian library. Correction improves object geometry and organizes cluttered scenes.">
</p>

## Code and checkpoints

**Code and trained checkpoints are coming soon.** This repository currently contains the paper overview and results. We will add implementations, sampling instructions, and checkpoints here as they become ready.

## Citation

```bibtex
@misc{ahn2026correction,
  title  = {Correction Training for Diffusion Models},
  author = {Ahn, Donghoon and Wang, Runqian and Efros, Alexei A. and Park, Taesung and Belhe, Yash},
  year   = {2026}
}
```

## Acknowledgments

Our experiments build on [DiT](https://github.com/facebookresearch/DiT), [SiT](https://github.com/willisma/SiT), [REPA](https://github.com/sihyun-yu/REPA), [LightningDiT](https://github.com/hustvl/LightningDiT), and [JiT](https://github.com/LTH14/JiT). We thank their authors for sharing their code and models.
