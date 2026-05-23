# AdaSteer Paper: Related Works Section

**Status:** Initial draft (May 23, 2026). Includes three organizational proposals, a recommendation, and a written draft using the recommended scheme.

---

## Three Organizational Proposals

### Proposal A: Method-axis organization

Group papers by the type of technique, broken down by what part of the pipeline they touch.

1. Video Diffusion Models and Transformers (the backbones AdaSteer adapts)
2. Test-Time Adaptation in Vision (the paradigm we extend to large video DiTs)
3. Parameter-Efficient Fine-Tuning (the standard adaptation tool family: LoRA, adapters, bias-tuning, spectral)
4. Adaptation and Personalization of Diffusion Models (DreamBooth-style and downstream subject/motion adaptation)
5. Modulation and Conditioning in Generative Models (FiLM → AdaLN → AdaLN-LoRA, i.e., AdaSteer's substrate)
6. Evaluation of Video Generation (FVD, VBench, dataset benchmarks)

**Pros:** Easy to map each cited method to a concrete technical category. Reviewers searching for "where does this paper sit in TTA vs PEFT vs modulation" find everything quickly.

**Cons:** Less narrative pull; reads like a checklist.

### Proposal B: Problem-axis organization

Group papers by the problem being solved, regardless of technique.

1. Generative Video Models and Long-Horizon Continuation (what we are improving)
2. Specializing a Pretrained Generator to a Test Instance (personalization, customization, TTT, TTA on generative models)
3. Reducing Trainable Capacity for Single-Instance Adaptation (PEFT, spectral, bias-tuning)
4. Steering vs Retraining (modulation pathways, conditioning, control adapters)
5. Measuring Video Quality and Adaptation Effects (FVD, content bias, VBench)

**Pros:** Strongly tied to motivation; each subsection answers "why is this paper relevant to our paper?"

**Cons:** Some papers cut across multiple subsections (LoRA appears in 3 and arguably 2). Harder to keep clean.

### Proposal C: Hybrid / Narrative-aligned organization (Recommended)

Follow the same conceptual arc as the introduction: large video DiTs → why test-time adaptation is hard for them → how prior PEFT and personalization methods would solve it → how modulation pathways enable a far smaller adaptation surface → how we evaluate.

1. **Video Diffusion Transformers and Long-Horizon Continuation.** Define the model family AdaSteer operates on.
2. **Test-Time Adaptation in Vision.** Establish the TTA paradigm we extend; show that almost all prior TTA targets discriminative models.
3. **Adaptation and Personalization of Pretrained Diffusion Models.** Cover DreamBooth/Textual Inversion/HyperDreamBooth/CustomTTT — the work most directly comparable to ours in goal.
4. **Parameter-Efficient Fine-Tuning.** LoRA, Houlsby adapters, BitFit, SVDiff — the toolbox we compare against and from which TinyLoRA draws.
5. **Modulation Pathways and Adaptive Normalization.** FiLM, AdaLN-Zero in DiT, AdaLN-LoRA, etc. This is the substrate AdaSteer reuses.
6. **Evaluation of Video Generation.** FVD, content bias, VBench, plus dataset papers (Panda-70M, UCF-101).

**Pros:** Reads like the rest of the paper, makes the contribution feel inevitable, and naturally surfaces the white space the method fills. Each subsection ends with a "gap" sentence that motivates AdaSteer.

**Cons:** Slightly longer than strictly necessary; if we are space-constrained for the camera-ready, we may need to fold (2) and (3) together.

**Recommendation: Proposal C.** It mirrors the paper's narrative, makes the gap explicit, and gives reviewers a clear way to compare AdaSteer against the closest prior work in each category. For a workshop paper or short conference paper, we can collapse to 4 subsections by merging (1)+(6) into "Background" and (2)+(3) into "Adapting Pretrained Generators."

---

## Citation Inclusion Policy

All cited papers below have a working URL. The selection rule we followed:

- Default to papers from the last 5 years (2021-2026) at reputable venues: CVPR, ICCV, ECCV, NeurIPS, ICML, ICLR.
- Older papers are included only when they are:
  - Foundational (FiLM, Houlsby adapters, DDPM, UCF-101): too commonly used to omit even if older than 5 years.
  - Directly tied to AdaSteer's mechanism or evaluation (FiLM for modulation, FVD/UCF-101 for evaluation continuity).
- We exclude technical reports without peer-review only when a peer-reviewed equivalent exists; LongCat-Video, CogVideoX, HunyuanVideo, and SVD are kept because they are the actual modern systems the field compares against.
- All cited dataset/benchmark/evaluation papers (FVD, Panda-70M, UCF-101, VBench) are required for reproducibility and methodological honesty.

---

## Justification for Older-than-5-Years Inclusions

| Paper | Year | Justification |
|---|---|---|
| FiLM (Perez et al.) | 2018 | Foundational ancestor of all feature-wise affine modulation methods, including AdaLN-Zero in DiT, which is AdaSteer's substrate. Almost every modern diffusion-transformer paper traces its modulation design to FiLM. |
| Houlsby Adapters | 2019 | Foundational adapter paper. Cited by every PEFT survey and by LoRA itself. Establishes the "small bottleneck inserted at frozen layers" paradigm that AdaSteer departs from. |
| DDPM (Ho et al.) | 2020 | Foundational diffusion paper. Required to ground the denoising objective AdaSteer optimizes. |
| TTT (Sun et al.) | 2020 | The original test-time training framework. AdaSteer is structurally a generative-model TTT method; we cannot omit the canonical reference. |
| Tent (Wang et al.) | 2021 | The canonical fully test-time adaptation paper; its "adapt only channel-wise affine parameters" framing is closely parallel to our argument for adapting only a shared timestep residual. |
| UCF-101 (Soomro et al.) | 2012 | Standard video benchmark used by every video generation evaluation including ours. We must cite for reproducibility; there is no modern replacement at the same evaluation scale. |

---

## Draft Related Works Section (using Proposal C)

### 2. Related Work

#### 2.1 Video Diffusion Transformers and Long-Horizon Continuation

Diffusion models have become the dominant class of generative models for high-resolution visual synthesis, with denoising diffusion probabilistic models providing the canonical training objective ([Ho et al., 2020](https://arxiv.org/abs/2006.11239)) and latent diffusion bringing the cost of high-resolution synthesis into the range of practical training budgets ([Rombach et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)). Replacing the U-Net backbone with a transformer ([Peebles & Xie, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html)) demonstrated that diffusion models scale predictably with transformer width, depth, and tokens, and introduced AdaLN-Zero, the per-block adaptive layer-norm modulation that AdaSteer directly reuses.

The video setting introduces two pressures: high token counts from the temporal dimension, and the need for long-horizon temporal coherence. Stable Video Diffusion ([Blattmann et al., 2023](https://arxiv.org/abs/2311.15127)) extends latent diffusion to image-to-video with curated pretraining stages. CogVideoX ([Yang et al., 2024](https://arxiv.org/abs/2408.06072)) and HunyuanVideo ([Kong et al., 2024](https://arxiv.org/abs/2412.03603)) push the diffusion-transformer formulation to 5B-13B parameters with full 3D attention and expert/dual-stream layouts. Stable Diffusion 3 / MMDiT ([Esser et al., ICML 2024](https://arxiv.org/abs/2403.03206)) couples the DiT architecture with rectified-flow training ([Lipman et al., ICLR 2023](https://openreview.net/forum?id=PqvMRDCJT9t)), and this combination is the formulation adopted by our backbone. The most relevant system for this work is LongCat-Video ([Meituan LongCat Team, 2025](https://arxiv.org/abs/2510.22200)), a 13.6B-parameter DiT trained with rectified flow that explicitly supports text-to-video, image-to-video, and video-continuation in a single set of weights and is pretrained on minute-scale continuation. We use LongCat-Video as our backbone precisely because video-continuation is its native training regime, which makes the conditioning frames a meaningful adaptation signal at inference.

These models all share two architectural features that AdaSteer's design depends on: (i) a single global timestep embedding pathway, and (ii) per-block adaLN modulation layers that re-project that pathway into shift, scale, and gate vectors. Despite this shared structure, none of these works expose the modulation pathway as a *test-time* adaptation interface; all adaptation is left to fine-tuning, LoRA, or full retraining. This is the gap AdaSteer fills.

#### 2.2 Test-Time Adaptation in Vision

Test-time adaptation (TTA) and test-time training (TTT) form a closely related family of methods that update a pretrained model on each test instance or batch before producing predictions. TTT ([Sun et al., ICML 2020](https://proceedings.mlr.press/v119/sun20b.html)) turns each unlabeled test sample into a self-supervised problem (e.g., rotation prediction) and updates the model briefly before predicting. Tent ([Wang et al., ICLR 2021](https://openreview.net/forum?id=uXl3bZLkr3c)) shows that on classification tasks, a fully test-time adaptation that updates only channel-wise affine parameters by minimizing prediction entropy is sufficient to absorb large distribution shifts. A recent survey ([Xiao et al., 2024](https://arxiv.org/abs/2411.03687)) catalogs more than 400 follow-ups across model-update, inference-time, normalization, sample, and prompt-based TTA variants.

Two features of this literature are important for AdaSteer. First, the consistent finding that *very small, structured* adaptation surfaces (BN statistics, channel-wise affine parameters, a handful of feature-wise biases) often perform as well as or better than freeing larger parts of the network. AdaSteer extends this finding from discriminative models to a 13.6B-parameter video DiT by adapting only a 512-dimensional residual on the timestep embedding. Second, almost all prior TTA work is built on top of discriminative classifiers or segmentation models; very little extends to large generative models, and even less to video. The TTA-for-generation entries we are aware of cluster around customization (Section 2.3) and test-time *search* over noise trajectories ([Liu et al., 2025 — Video-T1](https://arxiv.org/abs/2503.18942)), which is orthogonal to weight/embedding adaptation. AdaSteer is, to our knowledge, the first TTA method that explicitly targets the modulation pathway of a billion-scale video DiT during single-clip continuation.

#### 2.3 Adaptation and Personalization of Pretrained Diffusion Models

A second body of work specializes a pretrained diffusion generator to specific subjects, styles, or motions, typically with a small set of reference images or a single reference video. DreamBooth ([Ruiz et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Ruiz_DreamBooth_Fine_Tuning_Text-to-Image_Diffusion_Models_for_Subject-Driven_Generation_CVPR_2023_paper.html)) fine-tunes the full text-to-image diffusion network with a class-prior preservation loss to bind a new token to a subject. Textual Inversion ([Gal et al., ICLR 2023](https://arxiv.org/abs/2208.01618)) takes the opposite extreme, learning only a new text embedding while keeping the diffusion network frozen. HyperDreamBooth ([Ruiz et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Ruiz_HyperDreamBooth_HyperNetworks_for_Fast_Personalization_of_Text-to-Image_Models_CVPR_2024_paper.pdf)) accelerates per-subject adaptation roughly 25x by predicting an initial low-rank weight update with a hypernetwork before a short fast-finetune.

In the video setting, CustomTTT ([Bi et al., AAAI 2025](https://arxiv.org/abs/2412.15646)) combines per-LoRA appearance and motion customization with an additional test-time training pass to repair artifacts that appear when multiple LoRAs are composed. The recent "One-Minute Video Generation with Test-Time Training" ([Dalal et al., CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Dalal_One-Minute_Video_Generation_with_Test-Time_Training_CVPR_2025_paper.pdf)) inserts TTT layers into a CogVideoX backbone so that hidden states themselves are small neural networks, enabling long-context generation that linear-attention alternatives cannot match.

Both of these are useful conceptual neighbors. They share with AdaSteer the goal of specializing a frozen pretrained video diffusion transformer at inference time. They differ in two important ways. (i) Their adaptation surface is large: per-LoRA modules in CustomTTT add tens to hundreds of thousands of parameters, and TTT layers add new neural-network hidden states. AdaSteer adapts only a single 512-dimensional residual that is shared across all 48 blocks. (ii) Their objective is style/motion *transfer* from a reference, not steering the *same* test video's own conditioning frames. The closest setting to AdaSteer's is video continuation, where the conditioning frames themselves contain the only available adaptation signal.

#### 2.4 Parameter-Efficient Fine-Tuning

Parameter-efficient fine-tuning (PEFT) provides the standard family of adaptation tools used in the diffusion ecosystem. Houlsby-style adapters ([Houlsby et al., ICML 2019](https://proceedings.mlr.press/v97/houlsby19a/houlsby19a.pdf)) insert small bottleneck modules between frozen transformer sublayers. LoRA ([Hu et al., ICLR 2022](https://arxiv.org/abs/2106.09685)) instead learns a low-rank additive update to selected linear weights, which can be merged back into the base model at deployment. BitFit ([Zaken et al., ACL 2022](https://arxiv.org/abs/2106.10199)) shows that updating only a model's bias terms — under 0.1% of parameters — can match full fine-tuning on language tasks. SVDiff ([Han et al., ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Han_SVDiff_Compact_Parameter_Space_for_Diffusion_Fine-Tuning_ICCV_2023_paper.html)) is the closest cousin of the TinyLoRA variant we report: it fine-tunes only the singular values of each weight matrix's SVD, yielding compact adaptation footprints of order a few megabytes for text-to-image personalization.

Two empirical patterns emerge from this literature on large diffusion backbones. First, the rank or dimensionality of the adaptation subspace can be remarkably small and still produce useful adaptations. Second, when the adaptation signal is itself small — a single clip rather than a curated multi-image set — LoRA-style methods can either fail to move the model in a useful direction or overfit within a small number of gradient steps. Our own LoRA sweeps in this paper reproduce both failure modes on LongCat-Video. AdaSteer attacks the problem from a different direction: rather than choosing a smaller free subspace, it reuses the *frozen* per-block adaLN projections as a structured de-tying mechanism that turns a tiny shared residual into block-specific modulation, eliminating most of the degrees of freedom that LoRA leaves open.

#### 2.5 Modulation Pathways and Adaptive Normalization

The conceptual core of AdaSteer is that pretrained adaptive-normalization pathways can be reused as an adaptation interface. Feature-wise Linear Modulation ([Perez et al., AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11671)) introduces the general per-channel affine scale-and-shift conditioning operation that all subsequent adaptive-normalization variants inherit. We cite FiLM despite its age because it remains the cleanest formal statement of the modulation mechanism we exploit. Conditional batch and layer normalization in diffusion models follow the same template, and AdaLN-Zero in DiT ([Peebles & Xie, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html)) makes the timestep embedding the primary modulator: each transformer block linearly projects a global conditioning vector into shift/scale/gate vectors for both attention and the MLP. CogVideoX and HunyuanVideo retain this AdaLN-style modulation; LongCat-Video uses it as well, exposing a single 512-dimensional timestep embedding shared across 48 transformer blocks.

A small but growing line of work observes that the modulation pathway itself can be made more efficient or more controllable. AdaLN-LoRA, used inside Cosmos-style video DiTs, applies a low-rank adapter to the adaLN projection. AdaSteer is the test-time, per-video analogue of this idea: instead of training new low-rank weights inside the modulation projection, we leave the frozen modulation projection alone and add a small shared residual *upstream* of it, on the timestep embedding. The frozen per-block adaLN matrices then automatically translate that shared residual into block-specific shift/scale/gate perturbations. This is what we call structured weight tying — the trainable surface is global, but the induced perturbation is per-block.

#### 2.6 Evaluation of Video Generation

We adopt the standard combination of distributional and per-frame metrics. Fréchet Video Distance ([Unterthiner et al., 2018](https://arxiv.org/abs/1812.01717)) embeds videos through an Inflated 3D ConvNet and computes a Fréchet distance between real and generated activations; it remains the most widely reported single-number video metric. Recent work shows that FVD is partly biased toward per-frame content quality and that swapping the I3D backbone for self-supervised features can mitigate this content bias ([Ge et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Ge_On_the_Content_Bias_in_Frechet_Video_Distance_CVPR_2024_paper.pdf)). VBench ([Huang et al., CVPR 2024](https://vchitect.github.io/VBench-project/)) decomposes video quality into a battery of fine-grained, human-aligned dimensions and is now the de facto benchmark for text-to-video systems; VBench-2.0 extends the suite toward physics, commonsense, and motion realism.

For datasets, we evaluate on Panda-70M ([Chen et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Panda-70M_Captioning_70M_Videos_with_Multiple_Cross-Modality_Teachers_CVPR_2024_paper.html)), a 70M-clip captioned video dataset built from HD-VILA-100M with multiple cross-modality teacher captions, and UCF-101 ([Soomro et al., 2012](https://arxiv.org/abs/1212.0402)), a standard 101-class action recognition dataset that has been adopted as a continuation benchmark by most prior video-generation evaluations. We include UCF-101 despite its age because doing so preserves comparability with the existing video-generation evaluation literature, and we follow the now-standard practice of reporting both FVD and per-frame PSNR/SSIM/LPIPS on UCF-101.

In our experimental section, we follow this evaluation surface and report FVD/FID for distributional quality together with PSNR, SSIM, and LPIPS for per-frame fidelity, keeping standard-horizon and long-horizon settings explicitly separated.

---

## Reference List with Working URLs

The following table consolidates the citations above for easy verification.

| Cite key | Venue | URL |
|---|---|---|
| Ho et al., 2020 (DDPM) | NeurIPS 2020 | https://arxiv.org/abs/2006.11239 |
| Rombach et al., 2022 (LDM) | CVPR 2022 | https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf |
| Peebles & Xie, 2023 (DiT, AdaLN-Zero) | ICCV 2023 | https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html |
| Lipman et al., 2023 (Flow Matching) | ICLR 2023 | https://openreview.net/forum?id=PqvMRDCJT9t |
| Blattmann et al., 2023 (SVD) | arXiv 2023 | https://arxiv.org/abs/2311.15127 |
| Yang et al., 2024 (CogVideoX) | arXiv 2024 | https://arxiv.org/abs/2408.06072 |
| Kong et al., 2024 (HunyuanVideo) | arXiv 2024 | https://arxiv.org/abs/2412.03603 |
| Esser et al., 2024 (SD3 / MMDiT) | ICML 2024 | https://arxiv.org/abs/2403.03206 |
| Meituan LongCat Team, 2025 (LongCat-Video) | arXiv tech report 2025 | https://arxiv.org/abs/2510.22200 |
| Sun et al., 2020 (TTT) | ICML 2020 | https://proceedings.mlr.press/v119/sun20b.html |
| Wang et al., 2021 (Tent) | ICLR 2021 | https://openreview.net/forum?id=uXl3bZLkr3c |
| Xiao et al., 2024 (TTA survey) | arXiv 2024 | https://arxiv.org/abs/2411.03687 |
| Bi et al., 2025 (CustomTTT) | AAAI 2025 | https://arxiv.org/abs/2412.15646 |
| Dalal et al., 2025 (One-Minute Video with TTT) | CVPR 2025 | https://openaccess.thecvf.com/content/CVPR2025/papers/Dalal_One-Minute_Video_Generation_with_Test-Time_Training_CVPR_2025_paper.pdf |
| Liu et al., 2025 (Video-T1) | arXiv 2025 | https://arxiv.org/abs/2503.18942 |
| Ruiz et al., 2023 (DreamBooth) | CVPR 2023 | https://openaccess.thecvf.com/content/CVPR2023/html/Ruiz_DreamBooth_Fine_Tuning_Text-to-Image_Diffusion_Models_for_Subject-Driven_Generation_CVPR_2023_paper.html |
| Ruiz et al., 2024 (HyperDreamBooth) | CVPR 2024 | https://openaccess.thecvf.com/content/CVPR2024/papers/Ruiz_HyperDreamBooth_HyperNetworks_for_Fast_Personalization_of_Text-to-Image_Models_CVPR_2024_paper.pdf |
| Gal et al., 2023 (Textual Inversion) | ICLR 2023 | https://arxiv.org/abs/2208.01618 |
| Houlsby et al., 2019 (Adapters) | ICML 2019 | https://proceedings.mlr.press/v97/houlsby19a/houlsby19a.pdf |
| Hu et al., 2022 (LoRA) | ICLR 2022 | https://arxiv.org/abs/2106.09685 |
| Zaken et al., 2022 (BitFit) | ACL 2022 | https://arxiv.org/abs/2106.10199 |
| Han et al., 2023 (SVDiff) | ICCV 2023 | https://openaccess.thecvf.com/content/ICCV2023/html/Han_SVDiff_Compact_Parameter_Space_for_Diffusion_Fine-Tuning_ICCV_2023_paper.html |
| Perez et al., 2018 (FiLM) | AAAI 2018 | https://ojs.aaai.org/index.php/AAAI/article/view/11671 |
| Unterthiner et al., 2018 (FVD) | arXiv 2018 | https://arxiv.org/abs/1812.01717 |
| Ge et al., 2024 (Content-debiased FVD) | CVPR 2024 | https://content-debiased-fvd.github.io/ |
| Huang et al., 2024 (VBench) | CVPR 2024 | https://vchitect.github.io/VBench-project/ |
| Chen et al., 2024 (Panda-70M) | CVPR 2024 | https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Panda-70M_Captioning_70M_Videos_with_Multiple_Cross-Modality_Teachers_CVPR_2024_paper.html |
| Soomro et al., 2012 (UCF-101) | arXiv 2012 (CRCV-TR-12-01) | https://arxiv.org/abs/1212.0402 |

---

## Open Items for Your Review

1. **Pick an organizational proposal.** I recommend Proposal C; please confirm or pick A/B.
2. **TTT-vs-AdaSteer framing.** Section 2.2 currently positions AdaSteer as "TTA for generative video." If the paper later pivots to emphasize *steering* over *adaptation*, we should soften this and lift modulation/adaLN to be the primary related-work axis.
3. **Long-context Panda result framing.** Section 2.2's gap statement assumes our 1000-video Panda result is treated as a positive (FVD-improving) standard-horizon result and a long-horizon caveat. If we decide to present long-context Panda as a negative result in the main paper, the related work should explicitly acknowledge that prior long-context TTT work (Dalal et al., 2025) does not solve the same problem.
4. **Missing closely-related work?** Two areas I deliberately did not include but can add on request:
   - Video editing / control adapters (T2I-Adapter, ControlNet, VideoControlNet). These are conditioning-adapter methods, conceptually related but not test-time.
   - Diffusion guidance (CFG, autoguidance) as a "steering without weight update" baseline. Worth a paragraph if reviewers compare us to guidance variants.
5. **Citation style.** Above I used Markdown links for readability. For the LaTeX submission, I will translate each into `\citep{}` entries and produce a `.bib` file. If you prefer numbered IEEE style, say so now and I'll switch.
