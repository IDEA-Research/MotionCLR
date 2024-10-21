# MotionCLR: Motion Generation and Training-free Editing via Understanding Attention Mechanisms

[Ling-Hao Chen](https://lhchen.top/)$^*$, [Wenxun Dai](https://github.com/Dai-Wenxun), [Xuan Ju](https://juxuan27.github.io/), [Shunlin Lu](https://shunlinlu.github.io), [Lei Zhang](https://leizhang.org)†

$^*$Internship at IDEA Research. †Correspondence.

## 🤩 Abstract
> This research delves into analyzing the attention mechanism of diffusion models in human motion generation. Previous motion diffusion models lack explicit modeling of the word-level text-motion correspondence and explainability. Regarding these issues, we propose an attention-based motion diffusion model, namely MotionCLR, with CLeaR modeling of attention mechanisms. Based on the proposed model, we thoroughly analyze the formulation of the attention mechanism theoretically and empirically. Importantly, we highlight that the self-attention mechanism works to find the fine-grained word-sequence correspondence and activate the corresponding timesteps in the motion sequence. Besides, the cross-attention mechanism aims to measure the sequential similarity between frames and order the sequentiality of motion features. Motivated by these key insights, we propose versatile simple yet effective motion editing methods via manipulating attention maps, such as motion (de)-emphasizing, in-place motion replacement, and example-based motion generation \etc. For further verification of the explainability of the attention mechanism, we additionally explore the potential of action-counting and grounded motion generation ability via attention maps.

