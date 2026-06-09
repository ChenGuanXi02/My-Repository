import sys

import torch
from utils.train_utils import set_random_seed

from utils import init_env
import os
import argparse
from pathlib import Path

from utils.collate_utils import collate
from utils.import_utils import instantiate_from_config, recurse_instantiate_from_config, get_obj_from_str
from utils.init_utils import add_args
from torch.utils.data import DataLoader
from utils.trainer import Trainer

# >>> VIS PRIORS BEGIN: imports
from utils.vis_priors import get_unet_and_decoder, register_decoder_taps, agg_norm, save_strip
import numpy as np
# >>> VIS PRIORS END: imports


set_random_seed(7)

# # 2w预训练!!!
def get_loader(cfg):
    # 实例化
    casiav1_test_dataset = instantiate_from_config(cfg.test_dataset.CASIAv1)
    # nist16_test_dataset = instantiate_from_config(cfg.test_dataset.NIST16)
    # columbia_test_dataset = instantiate_from_config(cfg.test_dataset.Columbia)
    # coverage_test_dataset = instantiate_from_config(cfg.test_dataset.Coverage)
    # imd20_test_dataset = instantiate_from_config(cfg.test_dataset.IMD20)

    # 创建测试数据加载器
    casiav1_test_loader = DataLoader(
        casiav1_test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate
    )
    # nist16_test_loader = DataLoader(
    #     nist16_test_dataset,
    #     batch_size=cfg.batch_size,
    #     collate_fn=collate
    # )
    # columbia_test_loader = DataLoader(
    #     columbia_test_dataset,
    #     batch_size=cfg.batch_size,
    #     collate_fn=collate
    # )
    # coverage_test_loader = DataLoader(
    #     coverage_test_dataset,
    #     batch_size=cfg.batch_size,
    #     collate_fn=collate
    # )
    # imd20_test_loader = DataLoader(
    #     imd20_test_dataset,
    #     batch_size=cfg.batch_size,
    #     collate_fn=collate
    # )

    return casiav1_test_loader
    # casiav1_test_loader, nist16_test_loader
    # columbia_test_loader, coverage_test_loader, imd20_test_loader

# # fine_CASIA
# def get_loader(cfg):
#     CASIAv1_test_dataset = instantiate_from_config(cfg.test_dataset.CASIAv1)
#     CASIAv1_test_loader = DataLoader(
#              CASIAv1_test_dataset,
#              batch_size=cfg.batch_size,
#              collate_fn=collate
#          )
#     return CASIAv1_test_loader
#
# # fine_Cover
# def get_loader(cfg):
#     Cover_25_test_dataset = instantiate_from_config(cfg.test_dataset.Coverage_25)
#     Cover_25_test_dataset = DataLoader(
#              Cover_25_test_dataset,
#              batch_size=cfg.batch_size,
#              collate_fn=collate
#          )
#     return Cover_25_test_dataset
#
# # fine_NIST
# def get_loader(cfg):
#     NIST_160_test_dataset = instantiate_from_config(cfg.test_dataset.NIST16_160)
#     NIST_160_test_dataset = DataLoader(
#              NIST_160_test_dataset,
#              batch_size=cfg.batch_size,
#              collate_fn=collate
#          )
#     return NIST_160_test_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # >>> VIS PRIORS BEGIN: cli args
    parser.add_argument("--vis_priors", action="store_true",
                        help="Save prior-fusion heatmaps (Fg/Fl/Fc/c) during inference")
    parser.add_argument("--vis_n", type=int, default=4,
                        help="Number of samples to visualize")
    parser.add_argument("--vis_dir", type=str, default="vis/priors",
                        help="Directory to save visualization strips")
    # >>> VIS PRIORS END: cli args

    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--results_folder', type=str, default='./results')
    parser.add_argument('--num_epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulate_every', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_sample_steps', type=int, default=None)

    parser.add_argument('--target_dataset', nargs='+', type=str,
                        default=['CASIAv1'])
    # CASIAv1 Columbia Coverage IMD20 NIST16
    # CASIAv1 Coverage_25 NIST16_160

    parser.add_argument('--time_ensemble', action='store_true')
    parser.add_argument('--batch_ensemble', action='store_true')

    cfg = add_args(parser)
    assert not (cfg.time_ensemble and cfg.batch_ensemble), 'Cannot use both time_ensemble and batch_ensemble'
    """
        Hack config here.
    """
    if cfg.num_sample_steps is not None:
        cfg.diffusion_model.params.num_sample_steps = cfg.num_sample_steps

    # CASIAv1 Columbia Coverage IMD20 NIST16
    # CASIAv1 Coverage_25 NIST16_160
    CASIAv1_test_loader = get_loader(cfg)

    cond_uvit = instantiate_from_config(cfg.cond_uvit,
                                        conditioning_klass=get_obj_from_str(cfg.cond_uvit.params.conditioning_klass))
    model = recurse_instantiate_from_config(cfg.model,
                                            unet=cond_uvit)

    diffusion_model = instantiate_from_config(cfg.diffusion_model,
                                              model=model)

    optimizer = instantiate_from_config(cfg.optimizer, params=model.parameters())

    trainer = Trainer(
        diffusion_model,
        train_loader=None, test_loader=None,
        train_val_forward_fn=get_obj_from_str(cfg.train_val_forward_fn),
        gradient_accumulate_every=cfg.gradient_accumulate_every,
        results_folder=cfg.results_folder,
        optimizer=optimizer,
        train_num_epoch=cfg.num_epoch,
        amp=cfg.fp16,
        # log_with=None,
        cfg=cfg,
    )

    trainer.load(pretrained_path=cfg.checkpoint)
    # CASIAv1 Columbia Coverage IMD20 NIST16
    # CASIAv1 Coverage_25 NIST16_160
    CASIAv1_test_loader = trainer.accelerator.prepare(CASIAv1_test_loader)

    dataset_map = {
        # 2w  预训练!!!
        'CASIAv1': CASIAv1_test_loader,
        # 'NIST16': NIST16_test_loader,
        # 'Coverage': Coverage_test_loader,
        # 'IMD20': IMD20_test_loader,
        # 'Columbia': Columbia_test_loader,

        # 微调
        # 'CASIAv1': CASIAv1_test_loader,
        # 'Coverage_25': Coverage_25_test_loader,
        # 'NIST16_160': NIST16_160_test_loader,
    }
    assert all([d_name in dataset_map.keys() for d_name in cfg.target_dataset]), \
        f'Invalid dataset name. Available dataset: {dataset_map.keys()}' \
        f'Your input: {cfg.target_dataset}'
    target_dataset = [(dataset_map[dataset_name], dataset_name) for dataset_name in cfg.target_dataset]

    for dataset, dataset_name in target_dataset:
        trainer.model.eval()
        # CASIAv1 Columbia Coverage IMD20 NIST16
        # CASIAv1 Coverage_25 NIST16_160
        mask_path = Path(cfg.test_dataset.CASIAv1.params.image_root).parent.parent

        save_to = Path(cfg.results_folder) / dataset_name
        os.makedirs(save_to, exist_ok=True)

        # >>> VIS PRIORS BEGIN: visualize a few samples BEFORE evaluation
        if cfg.vis_priors and trainer.accelerator.is_main_process:
            try:
                unet, decoder = get_unet_and_decoder(trainer)  # 从 diffusion wrapper 中定位 net 与 decode_head
            except Exception as e:
                print(f"[VIS] locate unet/decoder failed: {e}")
                unet = None;
                decoder = None

            if unet is not None and decoder is not None:
                taps, hook_handles = register_decoder_taps(decoder)
                device = trainer.accelerator.device
                trainer.model.eval()


                def _pick_image(batch):
                    # 尝试常见键名；否则挑第一个 [B,*,H,W] 的张量
                    if isinstance(batch, dict):
                        for k in ["image", "img", "I", "cond_img", "input"]:
                            if k in batch and torch.is_tensor(batch[k]) and batch[k].dim() == 4:
                                return batch[k]
                        for v in batch.values():
                            if torch.is_tensor(v) and v.dim() == 4: return v
                    elif isinstance(batch, (list, tuple)):
                        for v in batch:
                            if torch.is_tensor(v) and v.dim() == 4: return v
                    raise ValueError("Cannot infer image tensor from batch.")


                vis_count = 0
                for batch in dataset:
                    try:
                        I = _pick_image(batch).to(device)  # [B,3,H,W] or [B,1,H,W], 0..1
                    except Exception as e:
                        print(f"[VIS] skip batch (no image tensor): {e}")
                        continue

                    B, _, H, W = I.shape
                    # 选一个中间时间步：用配置里的 num_sample_steps/2，fallback=10
                    T = getattr(cfg.diffusion_model.params, "num_sample_steps", 10) if hasattr(cfg,
                                                                                               "diffusion_model") else 10
                    t_scalar = max(1, int(T) // 2)
                    t = torch.full((B,), t_scalar, device=device, dtype=torch.long)
                    x_t = torch.randn(B, 1, H, W, device=device)  # 与 net.forward(x,t,I) 对齐

                    with torch.no_grad():
                        # net.py 已提供 sample_unet(x,t,I) -> forward；钩子会自动抓取中间张量
                        _ = unet.sample_unet(x_t, t, I)

                    # 取 4 个特征并规整到图像大小
                    H_Fg = agg_norm(taps["Fg"], (H, W), reduce="mean")
                    H_Fl = agg_norm(taps["Fl"], (H, W), reduce="mean")
                    H_Fc = agg_norm(taps["Fc"], (H, W), reduce="mean")
                    H_c = agg_norm(taps["c"], (H, W), reduce="mean")

                    img_uint8 = (I[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                    # 统一从 cfg 里取（add_args 已把 CLI 写回 cfg）
                    vis_dir = getattr(cfg, "vis_dir", "vis/priors")
                    os.makedirs(vis_dir, exist_ok=True)

                    save_path = os.path.join(vis_dir, f"{dataset_name}_strip_t{t_scalar}_{vis_count:03d}.png")
                    save_strip(save_path, img_uint8, H_Fg[0:1], H_Fl[0:1], H_Fc[0:1], H_c[0:1], pred01=None)
                    print(f"[VIS] saved -> {save_path}")

                    vis_count += 1
                    if vis_count >= getattr(cfg, "vis_n", 4):
                        break

                # 清理 hook
                if 'hook_handles' in locals():
                    for _h in hook_handles: _h.remove()
        # >>> VIS PRIORS END: visualize a few samples BEFORE evaluation


        if cfg.batch_ensemble:
            # mae-->auc
            auc, _ = trainer.val_batch_ensemble(model=trainer.model,
                                                test_data_loader=dataset,
                                                accelerator=trainer.accelerator,
                                                thresholding=False,
                                                save_to=save_to)
        elif cfg.time_ensemble:
            auc, _ = trainer.val_time_ensemble(model=trainer.model,
                                               test_data_loader=dataset,
                                               accelerator=trainer.accelerator,
                                               thresholding=False,
                                               save_to=save_to)
        else:
            auc, _ = trainer.val(model=trainer.model,
                                 test_data_loader=dataset,
                                 accelerator=trainer.accelerator,
                                 thresholding=False,
                                 save_to=save_to)
        trainer.accelerator.wait_for_everyone()
        trainer.accelerator.print(f'{dataset_name} auc: {auc}')

        if trainer.accelerator.is_main_process:
            from utils.eval import eval

            eval_score = eval(
                mask_path=mask_path,
                pred_path=cfg.results_folder,
                dataset_name=dataset_name)
        trainer.accelerator.wait_for_everyone()
