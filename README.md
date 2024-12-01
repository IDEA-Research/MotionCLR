# MotionCLR: Motion Generation and Training-free Editing via Understanding Attention Mechanisms

[Ling-Hao Chen](https://lhchen.top/)$^{* 1, 2}$, [Wenxun Dai](https://github.com/Dai-Wenxun)$^1$, [Xuan Ju](https://juxuan27.github.io/)$^3$, [Shunlin Lu](https://shunlinlu.github.io)$^4$, [Lei Zhang](https://leizhang.org)† $^2$

$^1$ THU, $^2$ IDEA, $^3$ CUHK, $^4$ CUHK-SZ

$^*$ Internship at IDEA Research. †Correspondence.


<p align="center">
  <a href='https://arxiv.org/abs/2410.18977'>
  <img src='https://img.shields.io/badge/Arxiv-2410.18977-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
  </a> 
  <a href='https://arxiv.org/pdf/2410.18977.pdf'>
  <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'>
  </a> 
  <a href='https://lhchen.top/MotionCLR'>
  <img src='https://img.shields.io/badge/Project-Page-%23df5b46?style=flat&logo=Google%20chrome&logoColor=%23df5b46'></a> 
  <a href='https://huggingface.co/blog/EvanTHU/motionclr-blog'>
    <img src='https://img.shields.io/badge/Blog-post-4EABE6?style=flat&logoColor=4EABE6'></a>
  <a href='https://github.com/IDEA-Research/MotionCLR'>
  <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a> 
  <a href='https://huggingface.co/spaces/EvanTHU/MotionCLR'>
  <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow'></a>
  <a href='https://github.com/IDEA-Research/MotionCLR'>
  <img src='https://img.shields.io/badge/gradio-demo-red.svg'>
  </a> 
  <a href='https://youtu.be/CQffPl7VI_c'>
  <img src='https://img.shields.io/badge/YouTube-Video-EA3323?style=flat&logo=youtube&logoColor=EA3323'></a>
  <a href='https://www.bilibili.com/video/BV1oQymYUEDX/'>
    <img src='https://img.shields.io/badge/Bilibili-Video-4EABE6?style=flat&logo=Bilibili&logoColor=4EABE6'></a>
  <a href='LICENSE'>
  <img src='https://img.shields.io/badge/License-IDEA-blue.svg'>
  </a> 
  <a href="" target='_blank'>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=IDEA-Research.MotionCLR&left_color=gray&right_color=%2342b983">
  </a> 
</p>

![Teaser](https://lhchen.top/MotionCLR/assets/img/teaser.png)

## 🤩 Abstract
> This research delves into analyzing the attention mechanism of diffusion models in human motion generation. Previous motion diffusion models lack explicit modeling of the word-level text-motion correspondence and explainability. Regarding these issues, we propose an attention-based motion diffusion model, namely MotionCLR, with CLeaR modeling of attention mechanisms. Based on the proposed model, we thoroughly analyze the formulation of the attention mechanism theoretically and empirically. Importantly, we highlight that the self-attention mechanism works to find the fine-grained word-sequence correspondence and activate the corresponding timesteps in the motion sequence. Besides, the cross-attention mechanism aims to measure the sequential similarity between frames and order the sequentiality of motion features. Motivated by these key insights, we propose versatile simple yet effective motion editing methods via manipulating attention maps, such as motion (de)-emphasizing, in-place motion replacement, and example-based motion generation *etc.*. For further verification of the explainability of the attention mechanism, we additionally explore the potential of action-counting and grounded motion generation ability via attention maps.

- [x] 📌 Due to some issues with latest gradio 5, MotionCLR v1-preview huggingface demo for motion editing will be supported next week.


## 📢 News

+ **[2024-11-014] MotionCLR v1-preview demo is released at [HuggingFace](https://huggingface.co/spaces/EvanTHU/MotionCLR).**
+ **[2024-10-25] Project, code, and paper are released.**


## ☕️ Preparation



<details>
<summary><b> Environment preparation </b></summary>

```bash
conda create python=3.10 --name motionclr
conda activate motionclr
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

</details>


<details>
<summary><b> Dependencies </b></summary>


If you have the `sudo` permission, install `ffmpeg` for visualizing stick figure (if not already installed):

```
sudo apt update
sudo apt install ffmpeg
ffmpeg -version  # check!
```

If you do not have the `sudo` permission to install it, please install it via `conda`: 

```
conda install conda-forge::ffmpeg
ffmpeg -version  # check!
```

Run the following command to install [`git-lfs`](https://git-lfs.com/):
```
conda install conda-forge::git-lfs
```

Run the script to download dependencies materials:

```
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```

</details>



<details>
<summary><b> Dataset preparation </b></summary>

Please refer to [HumanML3D](https://github.com/EricGuo5513/HumanML3D) for text-to-motion dataset setup. Copy the result dataset to our repository:
```
cp -r ../HumanML3D/HumanML3D ./datasets/humanml3d
```
Copy the mean/std file of HumanML3D into the `./data` path as `t2m_mean.npy` and `t2m_std.npy` respectively. The data structure should be:
```
./data
├── checkpoints
├── glove
├── HumanML3D
├── t2m_mean.npy
└── t2m_std.npy
```

The unofficial method of data preparation can be found in this [issue](https://github.com/Dai-Wenxun/MotionLCM/issues/6).

</details>





<details>
<summary><b> Pretrained Model </b></summary>

```python
from huggingface_hub import hf_hub_download

ckptdir = './checkpoints/t2m/release'
mean_path = hf_hub_download(
    repo_id="EvanTHU/MotionCLR",
    filename="meta/mean.npy",
    local_dir=ckptdir,
    local_dir_use_symlinks=False
)

std_path = hf_hub_download(
    repo_id="EvanTHU/MotionCLR",
    filename="meta/std.npy",
    local_dir=ckptdir,
    local_dir_use_symlinks=False
)

model_path = hf_hub_download(
    repo_id="EvanTHU/MotionCLR",
    filename="model/latest.tar",
    local_dir=ckptdir,
    local_dir_use_symlinks=False
)

opt_path = hf_hub_download(
    repo_id="EvanTHU/MotionCLR",
    filename="opt.txt",
    local_dir=ckptdir,
    local_dir_use_symlinks=False
)
```
The downloaded files will be saved in the `checkpoints/t2m/release/` directory as follows:
```
checkpoints/
└── t2m
    ├── release
    │   ├── meta
    │   │   ├── mean.npy
    │   │   └── std.npy
    │   ├── model
    │   │   └── latest.tar
    │   └── opt.txt
```
</details>


<details>
  <summary><b>  Folder Structure </b></summary>

After the whole setup pipeline, the folder structure will look like:

```
MotionCLR
└── data
    ├── glove
    │   ├── our_vab_data.npy
    │   ├── our_vab_idx.pkl
    │   └── out_vab_words.pkl
    ├── pretrained_models
    │   ├── t2m
    │   │   ├── text_mot_match
    │   │   │   └── model
    │   │   │       └── finest.tar
    │   │   └── length_est_bigru
    │   │       └── model
    │   │           └── finest.tar
    ├── HumanML3D
    │   ├── new_joint_vecs
    │   │   └── ...
    │   ├── new_joints
    │   │   └── ...
    │   ├── texts
    │   │   └── ...
    │   ├── Mean.npy
    │   ├── Std.npy
    │   ├── test.txt
    │   ├── train_val.txt
    │   ├── train.txt
    │   └── val.txt
    |── t2m_mean.npy
    |── t2m_std.npy
```

</details>



## 👨‍🏫 Quick Start

### Training

```bash
bash train.sh
``` 

### Testing for Evaluation

```bash
bash test.sh
``` 

### Generate Results from Text

Please replace `$EXP_DIR` with the experiment directory name.

+ Generate motion from a set of text prompts (`./assets/prompts-replace.txt`), each line is a prompt. (results will be saved in `./checkpoints/t2m/$EXP_DIR/samples_*/`)


    ```bash
    python -m scripts.generate --input_text ./assets/prompts-replace.txt \
    --motion_length 8 \
    --self_attention \
    --no_eff \
    --edit_mode \
    --opt_path ./checkpoints/t2m/$EXP_DIR/opt.txt
    ```
    <details>
    <summary><b> Explanation of the arguments </b></summary>

    - `--input_text`: the path to the text file containing prompts.
    
    - `--motion_length`: the length (s) of the generated motion.
    
    - `--self_attention`: use self-attention mechanism.
    
    - `--no_eff`: do not use efficient attention.

    - `--edit_mode`: enable editing mode.
  
    - `--opt_path`: the path to the trained models.
    
    </details>


+ Generate motion from a prompt. (results will be saved in `./checkpoints/t2m/$EXP_DIR/samples_*/`)

    ```bash
    python -m scripts.generate --text_prompt "a man jumps." --motion_length 8  --self_attention --no_eff --opt_path ./checkpoints/t2m/$EXP_DIR/opt.txt
    ```

    <details>
    <summary><b> Explanation of the arguments </b></summary>

    - `--text_prompt`: the text prompt.

    - `--motion_length`: the length (s) of the generated motion.

    - `--self_attention`: use self-attention mechanism.

    - `--no_eff`: do not use efficient attention.

    - `--opt_path`: the path to the trained models.

    - `--vis_attn`: visualize attention maps. (save in `./checkpoints/t2m/$EXP_DIR/vis_attn/`)
    </details>
    

<details>
<summary><b> Other arguments </b></summary>

- `--vis_attn`: visualize attention maps.
</details>




## 🔧 Downstream Editing Applications


<details>
<summary><b>Deploy the demo locally </b></summary>

Our project is supported by the latest Gradio 5, which provides a user-friendly interface for motion editing. The demo is available at [HuggingFace](https://huggingface.co/spaces/EvanTHU/MotionCLR). If you want to run the demo locally, please refer to the following instructions:

```bash
pip install gradio --upgrade
```

Launch the demo:
```python
python app.py
```
</details>



<details>
<summary><b>Interaction with commands</b></summary>

You can also use generate or edit the motion via command line. The command is the same as the generation command: 

```bash
    python -m scripts.generate --input_text ./assets/prompts-replace.txt \
    --motion_length 8 \
    --self_attention \
    --no_eff \
    --edit_mode \
    --opt_path ./checkpoints/t2m/$EXP_DIR/opt.txt
```

Besides, you also need to edit the configuration in `./options/edit.yaml` to specify the editing mode. The detailed clarification of the configuration can be found in the comment of the configuration file.
</details>










## 🌹 Acknowledgement

The author team would like to acknowledge [Dr. Jingbo Wang](https://wangjingbo1219.github.io/) from Shanghai AI Laboratory and [Dr. Xingyu Chen](https://seanchenxy.github.io/) from Peking University for his constructive suggestions and discussions on downstream applications. We also would like to acknowledge [Mr. Hongyang Li](https://lhy-hongyangli.github.io/) and [Mr. Zhenhua Yang](https://yeungchenwa.github.io/) from SCUT for their detailed discussion on some technical details and writing.  [Mr. Bohong Chen](https://github.com/RobinWitch) from ZJU also provided us with insightful feedback on the evaluation and the presentations.  We convey our thanks to all of them.

We would like to thank the authors of the following repositories for their excellent work: 
[HumanML3D](https://github.com/EricGuo5513/HumanML3D),
[UniMoCap](https://github.com/LinghaoChan/UniMoCap),
[joints2smpl](https://github.com/wangsen1312/joints2smpl),
[HumanTOMATO](https://github.com/IDEA-Research/HumanTOMATO),
[MotionLCM](https://github.com/Dai-Wenxun/MotionLCM),
[StableMoFusion](https://github.com/h-y1heng/StableMoFusion).

## 📜 Citation

If you find this work useful, please consider citing our paper:

```bash
@article{motionclr,
  title={MotionCLR: Motion Generation and Training-free Editing via Understanding Attention Mechanisms},
  author={Chen, Ling-Hao and Dai, Wenxun and Ju, Xuan and Lu, Shunlin and Zhang, Lei},
  journal={arxiv:2410.18977},
  year={2024}
}
```

## 📚 License

This code is distributed under an [IDEA LICENSE](LICENSE), which not allowed for commercial usage. Note that our code depends on other libraries and datasets which each have their own respective licenses that must also be followed.

