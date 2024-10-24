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
from box import Box


def yaml_to_box(yaml_file):
    with open(yaml_file, "r") as file:
        yaml_data = yaml.safe_load(file)
    return Box(yaml_data)


if __name__ == "__main__":
    parser = GenerateOptions()
    opt = parser.parse()
    set_seed(opt.seed)
    device_id = opt.gpu_id
    device = torch.device("cuda:%d" % device_id if torch.cuda.is_available() else "cpu")
    opt.device = device

    assert opt.dataset_name == "t2m" or "kit"

    # Using a text prompt for generation
    if opt.text_prompt != "":
        texts = [opt.text_prompt]
        opt.num_samples = 1
        motion_lens = [opt.motion_length * opt.fps]
    
    # Or using texts (in .txt file) for generation
    elif opt.input_text != "":
        with open(opt.input_text, "r") as fr:
            texts = [line.strip() for line in fr.readlines()]
        opt.num_samples = len(texts)
        if opt.input_lens != "":
            with open(opt.input_lens, "r") as fr:
                motion_lens = [int(line.strip()) for line in fr.readlines()]
            assert len(texts) == len(
                motion_lens
            ), f"Please ensure that the motion length in {opt.input_lens} corresponds to the text in {opt.input_text}."
        else:
            motion_lens = [opt.motion_length * opt.fps for _ in range(opt.num_samples)]
    
    # Or usining texts in dataset
    else:
        gen_datasetloader = get_dataset_loader(
            opt, opt.num_samples, mode="hml_gt", split="test"
        )
        texts, _, motion_lens = next(iter(gen_datasetloader))

    # edit mode
    if opt.edit_mode:
        edit_config = yaml_to_box("options/edit.yaml")
    else:
        edit_config = yaml_to_box("options/noedit.yaml")
    print(edit_config)

    ckpt_path = pjoin(opt.model_dir, opt.which_ckpt + ".tar")
    checkpoint = torch.load(ckpt_path,map_location={'cuda:0': str(device)})
    niter = checkpoint.get('total_it', 0)
    # make save dir
    out_path = opt.output_dir
    if out_path == "":
        out_path = pjoin(opt.save_root, "samples_iter{}_seed{}".format(niter, opt.seed))
        if opt.text_prompt != "":
            out_path += "_" + opt.text_prompt.replace(" ", "_").replace(".", "")
        elif opt.input_text != "":
            out_path += "_" + os.path.basename(opt.input_text).replace(
                ".txt", ""
            ).replace(" ", "_").replace(".", "")
    os.makedirs(out_path, exist_ok=True)

    # load model
    model = build_models(opt, edit_config=edit_config, out_path=out_path)
    niter = load_model_weights(model, ckpt_path, use_ema=not opt.no_ema)

    # Create a pipeline for generation in diffusion model framework
    pipeline = DiffusePipeline(
        opt=opt,
        model=model,
        diffuser_name=opt.diffuser_name,
        device=device,
        num_inference_steps=opt.num_inference_steps,
        torch_dtype=torch.float16,
    )

    # generate
    pred_motions, _ = pipeline.generate(
        texts, torch.LongTensor([int(x) for x in motion_lens])
    )

    # Convert the generated motion representaion into 3D joint coordinates and save as npy file
    npy_dir = pjoin(out_path, "joints_npy")
    root_dir = pjoin(out_path, "root_npy")
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(root_dir, exist_ok=True)
    print(f"saving results npy file (3d joints) to [{npy_dir}]")
    mean = np.load(pjoin(opt.meta_dir, "mean.npy"))
    std = np.load(pjoin(opt.meta_dir, "std.npy"))
    samples = []

    root_list = []
    for i, motion in enumerate(pred_motions):
        motion = motion.cpu().numpy() * std + mean
        np.save(pjoin(npy_dir, f"raw_{i:02}.npy"), motion)
        npy_name = f"{i:02}.npy"
        # 1. recover 3d joints representation by ik
        motion = recover_from_ric(torch.from_numpy(motion).float(), opt.joints_num)
        # 2. put on Floor (Y axis)
        floor_height = motion.min(dim=0)[0].min(dim=0)[0][1]
        motion[:, :, 1] -= floor_height
        motion = motion.numpy()
        # 3. remove jitter
        motion = motion_temporal_filter(motion, sigma=1)

        # save root trajectory (Y axis)
        root_trajectory = motion[:, 0, :]
        root_list.append(root_trajectory)
        np.save(pjoin(root_dir, f"root_{i:02}.npy"), root_trajectory)
        y = root_trajectory[:, 1]

        plt.figure()
        plt.plot(y)

        plt.legend()

        plt.title("Root Joint Trajectory")
        plt.xlabel("Frame")
        plt.ylabel("Position")

        plt.savefig("./root_trajectory_xyz.png")
        np.save(pjoin(npy_dir, npy_name), motion)
        samples.append(motion)

    root_list_res = np.concatenate(root_list, axis=0)
    np.save("root_list.npy", root_list_res)
    
    # save the text and length conditions used for this generation
    with open(pjoin(out_path, "results.txt"), "w") as fw:
        fw.write("\n".join(texts))
    with open(pjoin(out_path, "results_lens.txt"), "w") as fw:
        fw.write("\n".join([str(l) for l in motion_lens]))

    # skeletal animation visualization
    print(f"saving motion videos to [{out_path}]...")
    for i, title in enumerate(texts):
        motion = samples[i]
        fname = f"{i:02}.mp4"
        kinematic_tree = (
            paramUtil.t2m_kinematic_chain
            if (opt.dataset_name == "t2m")
            else paramUtil.kit_kinematic_chain
        )
        plot_3d_motion(
            pjoin(out_path, fname),
            kinematic_tree,
            motion,
            title=title,
            fps=opt.fps,
            radius=opt.radius,
        )
