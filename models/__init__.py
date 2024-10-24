from .unet import MotionCLR


__all__ = ["MotionCLR"]


def build_models(opt, edit_config=None, out_path=None):
    print("\nInitializing model ...")
    model = MotionCLR(
        input_feats=opt.dim_pose,
        text_latent_dim=opt.text_latent_dim,
        base_dim=opt.base_dim,
        dim_mults=opt.dim_mults,
        time_dim=opt.time_dim,
        adagn=not opt.no_adagn,
        zero=True,
        dropout=opt.dropout,
        no_eff=opt.no_eff,
        cond_mask_prob=getattr(opt, "cond_mask_prob", 0.0),
        self_attention=opt.self_attention,
        vis_attn=opt.vis_attn,
        edit_config=edit_config,
        out_path=out_path,
    )

    return model
