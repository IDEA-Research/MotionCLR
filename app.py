import gradio as gr
import sys
import os
import torch
import numpy as np
from os.path import join as pjoin
import utils.paramUtil as paramUtil
from utils.plot_script import *
from utils.utils import *
from utils.motion_process import recover_from_ric
from accelerate.utils import set_seed
from models.gaussian_diffusion import DiffusePipeline
from options.generate_options import GenerateOptions
from utils.model_load import load_model_weights
from motion_loader import get_dataset_loader
from models import build_models
import yaml
import time
from box import Box
import hashlib
from huggingface_hub import hf_hub_download

ckptdir = './checkpoints/t2m/release'
os.makedirs(ckptdir, exist_ok=True)


os.environ['GRADIO_TEMP_DIR']="temp"
os.environ['GRADIO_ALLOWED_PATHS']="temp"

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



os.makedirs("temp", exist_ok=True)

def generate_md5(input_string):
    # Encode the string and compute the MD5 hash
    md5_hash = hashlib.md5(input_string.encode())
    # Return the hexadecimal representation of the hash
    return md5_hash.hexdigest()

def set_all_use_to_false(data):
    for key, value in data.items():
        if isinstance(value, Box): 
            set_all_use_to_false(value)
        elif key == 'use': 
            data[key] = False     
    return data

def yaml_to_box(yaml_file):
    with open(yaml_file, 'r') as file:
        yaml_data = yaml.safe_load(file)
    
    return Box(yaml_data)

HEAD = ("""<div>
<div class="embed_hidden" style="text-align: center;">
    <h1>MotionCLR: Motion Generation and Training-free Editing via Understanding Attention Mechanisms</h1>
    <h2>MotionCLR v1-preview Demo</h2>
    <h3>
        <a href="https://lhchen.top" target="_blank" rel="noopener noreferrer">Ling-Hao Chen</a><sup>1, 2</sup>,
        <a href="https://https://github.com/Dai-Wenxun" target="_blank" rel="noopener noreferrer">Wenxun Dai</a><sup>1</sup>,
        <a href="https://juxuan27.github.io/" target="_blank" rel="noopener noreferrer">Xuan Ju</a><sup>3</sup>,
        <a href="https://shunlinlu.github.io" target="_blank" rel="noopener noreferrer">Shunlin Lu</a><sup>4</sup>,
        <a href="https://leizhang.org" target="_blank" rel="noopener noreferrer">Lei Zhang</a><sup>2 🤗</sup>
    </h3>
    <h3><sup>🤗</sup><i>Corresponding author.</i></h3>
    <h3>
        <sup>1</sup>THU &emsp;
        <sup>2</sup>IDEA Research &emsp;
        <sup>3</sup>CUHK  &emsp;
        <sup>4</sup>CUHK (SZ)
    </h3>
</div>
<div style="display:flex; gap: 0.3rem; justify-content: center; align-items: center;" align="center">
<a href='https://arxiv.org/abs/2410.18977'><img src='https://img.shields.io/badge/Arxiv-2405.20340-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a> 
<a href='https://arxiv.org/pdf/2410.18977.pdf'><img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'></a> 
<a href='https://lhchen.top/MotionCLR'><img src='https://img.shields.io/badge/Project-Page-%23df5b46?style=flat&logo=Google%20chrome&logoColor=%23df5b46'></a> 
<a href='https://huggingface.co/blog/EvanTHU/motionclr-blog'><img src='https://img.shields.io/badge/Blog-post-4EABE6?style=flat&logoColor=4EABE6'></a>
<a href='https://github.com/IDEA-Research/MotionCLR'><img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a> 
<a href='https://huggingface.co/spaces/EvanTHU/MotionCLR'><img src='https://img.shields.io/badge/gradio-demo-red.svg'></a> 
<a href='LICENSE'><img src='https://img.shields.io/badge/License-IDEA-blue.svg'></a> 
<a href="https://huggingface.co/spaces/EvanTHU/MotionCLR" target='_blank'><img src="https://visitor-badge.laobi.icu/badge?page_id=IDEA-Research.MotionCLR&left_color=gray&right_color=%2342b983"></a> 
</div>
</div>
""")


edit_config = yaml_to_box('options/edit.yaml')
CSS = """
.retrieved_video {
    position: relative;
    margin: 0;
    box-shadow: var(--block-shadow);
    border-width: var(--block-border-width);
    border-color: #000000;
    border-radius: var(--block-radius);
    background: var(--block-background-fill);
    width: 100%;
    line-height: var(--line-sm);
}
.contour_video {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    z-index: var(--layer-5);
    border-radius: var(--block-radius);
    background: var(--background-fill-primary);
    padding: 0 var(--size-6);
    max-height: var(--size-screen-h);
    overflow: hidden;
}
"""


def generate_video_from_text(text, opt, pipeline):
    global edit_config

    gr.Info("Loading Configurations...", duration = 3)
    model = build_models(opt, edit_config=edit_config)
    ckpt_path = pjoin(opt.model_dir, opt.which_ckpt + '.tar')  
    niter = load_model_weights(model, ckpt_path, use_ema=not opt.no_ema)

    pipeline = DiffusePipeline(
        opt = opt,
        model = model, 
        diffuser_name = opt.diffuser_name, 
        device=opt.device,
        num_inference_steps=opt.num_inference_steps,
        torch_dtype=torch.float16,
    )
    
    width = 500
    height = 500
    texts = [text, text]
    motion_lens = [opt.motion_length * opt.fps, opt.motion_length * opt.fps]
    
    save_dir = 'temp/'
    filename = generate_md5(str(time.time())) + ".gif"
    save_path = pjoin(save_dir, str(filename))
    os.makedirs(save_dir, exist_ok=True)
    
    print("xxxxxxx")
    print(texts)
    print(motion_lens)
    print("xxxxxxx")
    
    start_time = time.perf_counter()
    gr.Info("Generating motion...", duration = 3)
    pred_motions, _ = pipeline.generate(texts, torch.LongTensor([int(x) for x in motion_lens]))
    end_time = time.perf_counter()
    exc = end_time - start_time
    gr.Info(f"Generating time cost: {exc:.2f} s, rendering starts...", duration = 3)
    start_time = time.perf_counter()
    mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(opt.meta_dir, 'std.npy'))
    print(mean)
    print(std)
    print(pred_motions)
    
    
    samples = []
    
    root_list = []
    for i, motion in enumerate(pred_motions):
        motion = motion.cpu().numpy() * std + mean
        # 1. recover 3d joints representation by ik
        motion = recover_from_ric(torch.from_numpy(motion).float(), opt.joints_num)
        # 2. put on Floor (Y axis)
        floor_height = motion.min(dim=0)[0].min(dim=0)[0][1]
        motion[:, :, 1] -= floor_height
        motion = motion.numpy()
        # 3. remove jitter
        motion = motion_temporal_filter(motion, sigma=1)

        samples.append(motion)
    
    i = 1
    title = texts[i]
    motion = samples[i]
    kinematic_tree = paramUtil.t2m_kinematic_chain if (opt.dataset_name == 't2m') else paramUtil.kit_kinematic_chain
    plot_3d_motion(save_path, kinematic_tree, motion, title=title, fps=opt.fps, radius=opt.radius)


    gr.Info("Rendered motion...", duration = 3)
    end_time = time.perf_counter()
    exc = end_time - start_time
    gr.Info(f"Rendering time cost: {exc:.2f} s", duration = 3)
    
    video_dis = f'<img src="/gradio_api/file={save_path}" width="{width}" style="display: block; margin: 0 auto;">'
    style_dis = video_dis 
    return video_dis, style_dis, video_dis, gr.update(visible=True)


def reweighting(text, idx, weight, opt, pipeline):
    global edit_config
    edit_config.reweighting_attn.use = True
    edit_config.reweighting_attn.idx = idx
    edit_config.reweighting_attn.reweighting_attn_weight = weight


    gr.Info("Loading Configurations...", duration = 3)
    model = build_models(opt, edit_config=edit_config)
    ckpt_path = pjoin(opt.model_dir, opt.which_ckpt + '.tar')  
    niter = load_model_weights(model, ckpt_path, use_ema=not opt.no_ema)

    pipeline = DiffusePipeline(
        opt = opt,
        model = model, 
        diffuser_name = opt.diffuser_name, 
        device=opt.device,
        num_inference_steps=opt.num_inference_steps,
        torch_dtype=torch.float16,
    )
    
    print(edit_config)
    
    width = 500
    height = 500
    texts = [text, text]
    motion_lens = [opt.motion_length * opt.fps for _ in range(opt.num_samples)]
    
    save_dir = 'temp/'
    filenames = [generate_md5(str(time.time())) + ".gif", generate_md5(str(time.time())) + ".gif"]
    save_paths = [pjoin(save_dir, str(filenames[0])), pjoin(save_dir, str(filenames[1]))]
    os.makedirs(save_dir, exist_ok=True)
    
    start_time = time.perf_counter()
    gr.Info("Generating motion...", duration = 3)
    pred_motions, _ = pipeline.generate(texts, torch.LongTensor([int(x) for x in motion_lens]))
    end_time = time.perf_counter()
    exc = end_time - start_time
    gr.Info(f"Generating time cost: {exc:.2f} s, rendering starts...", duration = 3)
    start_time = time.perf_counter()
    mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(opt.meta_dir, 'std.npy'))
    
    
    samples = []
    
    root_list = []
    for i, motion in enumerate(pred_motions):
        motion = motion.cpu().numpy() * std + mean
        # 1. recover 3d joints representation by ik
        motion = recover_from_ric(torch.from_numpy(motion).float(), opt.joints_num)
        # 2. put on Floor (Y axis)
        floor_height = motion.min(dim=0)[0].min(dim=0)[0][1]
        motion[:, :, 1] -= floor_height
        motion = motion.numpy()
        # 3. remove jitter
        motion = motion_temporal_filter(motion, sigma=1)

        samples.append(motion)
    
    i = 1
    title = texts[i]
    motion = samples[i]
    kinematic_tree = paramUtil.t2m_kinematic_chain if (opt.dataset_name == 't2m') else paramUtil.kit_kinematic_chain
    plot_3d_motion(save_paths[1], kinematic_tree, motion, title=title, fps=opt.fps, radius=opt.radius)

    
    gr.Info("Rendered motion...", duration = 3)
    end_time = time.perf_counter()
    exc = end_time - start_time
    gr.Info(f"Rendering time cost: {exc:.2f} s", duration = 3)
    
    video_dis = f'<img width="{width}" style="display: block; margin: 0 auto;" src="/gradio_api/file={save_paths[1]}">'
    
    
    edit_config = set_all_use_to_false(edit_config)
    return video_dis


def generate_example_based_motion(text, chunk_size, example_based_steps_end, temp_seed, temp_seed_bar, num_motion, opt, pipeline):
    global edit_config
    edit_config.example_based.use = True
    edit_config.example_based.chunk_size = chunk_size
    edit_config.example_based.example_based_steps_end = example_based_steps_end
    edit_config.example_based.temp_seed = temp_seed
    edit_config.example_based.temp_seed_bar = temp_seed_bar


    gr.Info("Loading Configurations...", duration = 3)
    model = build_models(opt, edit_config=edit_config)
    ckpt_path = pjoin(opt.model_dir, opt.which_ckpt + '.tar')  
    niter = load_model_weights(model, ckpt_path, use_ema=not opt.no_ema)

    pipeline = DiffusePipeline(
        opt = opt,
        model = model, 
        diffuser_name = opt.diffuser_name, 
        device=opt.device,
        num_inference_steps=opt.num_inference_steps,
        torch_dtype=torch.float16,
    )
    
    width = 500
    height = 500
    texts = [text for _ in range(num_motion)]
    motion_lens = [opt.motion_length * opt.fps for _ in range(opt.num_samples)]
    
    save_dir = 'temp/'
    filenames = [generate_md5(str(time.time())) + ".gif" for _ in range(num_motion)]
    save_paths = [pjoin(save_dir, str(filenames[i])) for i in range(num_motion)]
    os.makedirs(save_dir, exist_ok=True)
    
    start_time = time.perf_counter()
    gr.Info("Generating motion...", duration = 3)
    pred_motions, _ = pipeline.generate(texts, torch.LongTensor([int(x) for x in motion_lens]))
    end_time = time.perf_counter()
    exc = end_time - start_time
    gr.Info(f"Generating time cost: {exc:.2f} s, rendering starts...", duration = 3)
    start_time = time.perf_counter()
    mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(opt.meta_dir, 'std.npy'))
    
    
    samples = []
    
    root_list = []
    progress=gr.Progress()
    progress(0, desc="Starting...")
    for i, motion in enumerate(pred_motions):
        motion = motion.cpu().numpy() * std + mean
        # 1. recover 3d joints representation by ik
        motion = recover_from_ric(torch.from_numpy(motion).float(), opt.joints_num)
        # 2. put on Floor (Y axis)
        floor_height = motion.min(dim=0)[0].min(dim=0)[0][1]
        motion[:, :, 1] -= floor_height
        motion = motion.numpy()
        # 3. remove jitter
        motion = motion_temporal_filter(motion, sigma=1)

        samples.append(motion)
    
    video_dis = []
    i = 0
    for title in progress.tqdm(texts):
        print(save_paths[i])
        title = texts[i]
        motion = samples[i]
        kinematic_tree = paramUtil.t2m_kinematic_chain if (opt.dataset_name == 't2m') else paramUtil.kit_kinematic_chain
        plot_3d_motion(save_paths[i], kinematic_tree, motion, title=title, fps=opt.fps, radius=opt.radius)
        video_html = f'''
        <img class="retrieved_video" width="{width}" height="{height}" preload="auto" src="/gradio_api/file={save_paths[i]}">
        '''
        video_dis.append(video_html)
        i += 1
        
    for _ in range(24 - num_motion):
        video_dis.append(None)
    gr.Info("Rendered motion...", duration = 3)
    end_time = time.perf_counter()
    exc = end_time - start_time
    gr.Info(f"Rendering time cost: {exc:.2f} s", duration = 3)
        
    edit_config = set_all_use_to_false(edit_config)
    return video_dis


def transfer_style(text, style_text, style_transfer_steps_end, opt, pipeline):
    global edit_config
    edit_config.style_tranfer.use = True
    edit_config.style_tranfer.style_transfer_steps_end = style_transfer_steps_end

    gr.Info("Loading Configurations...", duration = 3)
    model = build_models(opt, edit_config=edit_config)
    ckpt_path = pjoin(opt.model_dir, opt.which_ckpt + '.tar')  
    niter = load_model_weights(model, ckpt_path, use_ema=not opt.no_ema)

    pipeline = DiffusePipeline(
        opt = opt,
        model = model, 
        diffuser_name = opt.diffuser_name, 
        device=opt.device,
        num_inference_steps=opt.num_inference_steps,
        torch_dtype=torch.float16,
    )
    
    print(edit_config)
    
    width = 500
    height = 500
    texts = [style_text, text, text]
    motion_lens = [opt.motion_length * opt.fps for _ in range(opt.num_samples)]
    
    save_dir = 'temp/'
    filenames = [generate_md5(str(time.time())) + ".gif", generate_md5(str(time.time())) + ".gif", generate_md5(str(time.time())) + ".gif"]
    save_paths = [pjoin(save_dir, str(filenames[0])), pjoin(save_dir, str(filenames[1])), pjoin(save_dir, str(filenames[2]))]
    os.makedirs(save_dir, exist_ok=True)
    
    start_time = time.perf_counter()
    gr.Info("Generating motion...", duration = 3)
    pred_motions, _ = pipeline.generate(texts, torch.LongTensor([int(x) for x in motion_lens]))
    end_time = time.perf_counter()
    exc = end_time - start_time
    gr.Info(f"Generating time cost: {exc:.2f} s, rendering starts...", duration = 3)
    start_time = time.perf_counter()
    mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(opt.meta_dir, 'std.npy'))
    
    samples = []
    
    root_list = []
    for i, motion in enumerate(pred_motions):
        motion = motion.cpu().numpy() * std + mean
        # 1. recover 3d joints representation by ik
        motion = recover_from_ric(torch.from_numpy(motion).float(), opt.joints_num)
        # 2. put on Floor (Y axis)
        floor_height = motion.min(dim=0)[0].min(dim=0)[0][1]
        motion[:, :, 1] -= floor_height
        motion = motion.numpy()
        # 3. remove jitter
        motion = motion_temporal_filter(motion, sigma=1)

        samples.append(motion)
    
    for i,title in enumerate(texts):
        title = texts[i]
        motion = samples[i]
        kinematic_tree = paramUtil.t2m_kinematic_chain if (opt.dataset_name == 't2m') else paramUtil.kit_kinematic_chain
        plot_3d_motion(save_paths[i], kinematic_tree, motion, title=title, fps=opt.fps, radius=opt.radius)

    gr.Info("Rendered motion...", duration = 3)
    end_time = time.perf_counter()
    exc = end_time - start_time
    gr.Info(f"Rendering time cost: {exc:.2f} s", duration = 3)
    
    video_dis0 = f"""<img width="{width}" style="display: block; margin: 0 auto;" src="/gradio_api/file={save_paths[0]}"> <br> <p align="center"> Style Reference </p>"""
    video_dis1 = f"""<img width="{width}" style="display: block; margin: 0 auto;" src="/gradio_api/file={save_paths[2]}"> <br> <p align="center"> Content Reference </p>"""
    video_dis2 = f"""<img width="{width}" style="display: block; margin: 0 auto;" src="/gradio_api/file={save_paths[1]}"> <br> <p align="center"> Transfered Result </p>"""
     
    edit_config = set_all_use_to_false(edit_config)
    return video_dis0, video_dis2


def main():
    parser = GenerateOptions()
    opt = parser.parse_app()
    set_seed(opt.seed)
    device_id = opt.gpu_id
    device = torch.device('cuda:%d' % device_id if torch.cuda.is_available() else 'cpu')
    opt.device = device
    print(device)

    # load model
    model = build_models(opt, edit_config=edit_config)
    ckpt_path = pjoin(opt.model_dir, opt.which_ckpt + '.tar')  
    niter = load_model_weights(model, ckpt_path, use_ema=not opt.no_ema)

    pipeline = DiffusePipeline(
        opt = opt,
        model = model, 
        diffuser_name = opt.diffuser_name, 
        device=device,
        num_inference_steps=opt.num_inference_steps,
        torch_dtype=torch.float16,
    )
    
    with gr.Blocks(theme=gr.themes.Glass()) as demo:
        gr.HTML(HEAD)
        with gr.Row():
            with gr.Column(scale=7):
                text_input = gr.Textbox(label="Input the text prompt to generate motion...")
            with gr.Column(scale=3):
                sequence_length = gr.Slider(minimum=1, maximum=9.6, step=0.1, label="Motion length", value=8)
        with gr.Row(): 
            generate_button = gr.Button("Generate motion")
            
        with gr.Row():
            video_display = gr.HTML(label="Generated motion", visible=True)
        

        tabs = gr.Tabs(visible=False)
        with tabs:
            emph_tab = gr.Tab("Motion (de-)emphasizing", visible=False)
            with emph_tab:
                with gr.Row():
                    int_input = gr.Number(label="Editing word index", minimum=0, maximum=70)
                    weight_input = gr.Slider(minimum=-1, maximum=1, step=0.01, label="Input weight for (de-)emphasizing [-1, 1]", value=0)
                
                trim_button = gr.Button("Edit Motion")
                
                with gr.Row():
                    original_video1 = gr.HTML(label="before editing", visible=False)
                    edited_video = gr.HTML(label="after editing")
                
                trim_button.click(
                    fn=lambda x, int_input, weight_input : reweighting(x, int_input, weight_input, opt, pipeline), 
                    inputs=[text_input, int_input, weight_input],
                    outputs=edited_video,
                    )

            exp_tab = gr.Tab("Example-based motion genration", visible=False)
            with exp_tab:
                with gr.Row():
                    with gr.Column(scale=4):
                        chunk_size = gr.Number(minimum=10, maximum=20, step=10,label="Chunk size (#frames)", value=20)
                        example_based_steps_end = gr.Number(minimum=0, maximum=9,label="Ending step of manipulation", value=6)
                    with gr.Column(scale=3):
                        temp_seed = gr.Number(label="Seed for random", value=200, minimum=0)
                        temp_seed_bar = gr.Slider(minimum=0, maximum=100, step=1, label="Seed for random bar", value=15)
                    with gr.Column(scale=3):
                        num_motion = gr.Radio(choices=[4, 8, 12, 16, 24], value=8, label="Select number of motions")
                    
                gen_button = gr.Button("Generate example-based motion")
                
                
                example_video_display = []
                for _ in range(6):
                    with gr.Row():
                        for _ in range(4):
                            video = gr.HTML(label="Example-based motion", visible=True)
                            example_video_display.append(video)

                gen_button.click(
                    fn=lambda text, chunk_size, example_based_steps_end, temp_seed, temp_seed_bar, num_motion: generate_example_based_motion(text, chunk_size, example_based_steps_end, temp_seed, temp_seed_bar, num_motion, opt, pipeline),
                    inputs=[text_input, chunk_size, example_based_steps_end, temp_seed, temp_seed_bar, num_motion],
                    outputs=example_video_display
                )

            trans_tab = gr.Tab("Style transfer", visible=False)
            with trans_tab:
                with gr.Row():
                    style_text = gr.Textbox(label="Reference prompt (e.g. 'a man walks.')", value="a man walks.")
                    style_transfer_steps_end = gr.Number(label="The end step of diffusion (0~9)", minimum=0, maximum=9, value=5)

                style_transfer_button = gr.Button("Transfer style")

                with gr.Row():
                    style_reference = gr.HTML(label="style reference")
                    original_video4 = gr.HTML(label="before style transfer", visible=False)
                    styled_video = gr.HTML(label="after style transfer")

                style_transfer_button.click(
                    fn=lambda text, style_text, style_transfer_steps_end: transfer_style(text, style_text, style_transfer_steps_end, opt, pipeline),
                    inputs=[text_input, style_text, style_transfer_steps_end],
                    outputs=[style_reference, styled_video],
                )
        
        def update_motion_length(sequence_length):
            opt.motion_length = sequence_length
        
        def on_generate(text, length, pipeline):
            update_motion_length(length)
            return generate_video_from_text(text, opt, pipeline)

                
        generate_button.click(
            fn=lambda text, length: on_generate(text, length, pipeline),  
            inputs=[text_input, sequence_length],
            outputs=[
                video_display, 
                original_video1, 
                original_video4,
                tabs,
                ], 
            show_progress=True
        ).then(
            fn=lambda: [gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)],
            inputs=None,
            outputs=[video_display, original_video1, original_video4, emph_tab, exp_tab, trans_tab]
        )

    demo.launch()


if __name__ == '__main__':
    main()
