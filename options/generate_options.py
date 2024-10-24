import argparse
from .get_opt import get_opt

class GenerateOptions():
    def __init__(self, app=False):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize()

    def initialize(self):
        self.parser.add_argument("--opt_path", type=str, default='./checkpoints/t2m/t2m_condunet1d_batch64/opt.txt', help='option file path for loading model')
        self.parser.add_argument("--gpu_id", type=int, default=0, help='GPU id')
        self.parser.add_argument("--output_dir", type=str, default='', help='Directory path to save generation result')
        self.parser.add_argument("--footskate_cleanup", action="store_true", help='Where use footskate cleanup in inference')
        
        # inference
        self.parser.add_argument("--num_inference_steps", type=int, default=10, help='Number of iterative denoising steps during inference.')
        self.parser.add_argument("--which_ckpt", type=str, default='latest', help='name of checkpoint to load')
        self.parser.add_argument("--diffuser_name", type=str, default='dpmsolver', help='sampler\'s scheduler class name in the diffuser library')
        self.parser.add_argument("--no_ema", action="store_true", help='Where use EMA model in inference')
        self.parser.add_argument("--no_fp16", action="store_true", help='Whether use FP16 in inference')
        self.parser.add_argument('--batch_size', type=int, default=1, help='Batch size for generate')
        self.parser.add_argument("--seed", default=0, type=int, help="For fixing random seed.")
        self.parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
        
        # generate prompts
        self.parser.add_argument('--text_prompt', type=str, default="", help='One text description pompt for motion generation')
        self.parser.add_argument("--motion_length", default=4.0, type=float, help="The length of the generated motion [in seconds] when using prompts. Maximum is 9.8 for HumanML3D (text-to-motion)")
        self.parser.add_argument('--input_text', type=str, default='', help='File path of texts when using multiple texts.')
        self.parser.add_argument('--input_lens', type=str, default='', help='File path of expected motion frame lengths when using multitext.')
        self.parser.add_argument("--num_samples", type=int, default=10, help='Number of samples for generate when using dataset.')
        self.parser.add_argument('--debug', action="store_true", help='debug mode')
        self.parser.add_argument('--self_attention', action="store_true", help='self_attention use or not')
        self.parser.add_argument('--no_eff', action='store_true', help='whether use efficient linear attention')
        self.parser.add_argument('--vis_attn', action='store_true', help='vis attention value or not')
        self.parser.add_argument('--edit_mode', action='store_true', help='editing mode')


    def parse(self):
        self.opt = self.parser.parse_args()
        opt_path = self.opt.opt_path
        get_opt(self.opt, opt_path)
        return self.opt

    def parse_app(self):
        self.opt = self.parser.parse_args(
            args=['--motion_length', '8', '--self_attention', '--no_eff', '--opt_path', './checkpoints/t2m/release/opt.txt', '--edit_mode']
        )
        opt_path = self.opt.opt_path
        get_opt(self.opt, opt_path)
        return self.opt