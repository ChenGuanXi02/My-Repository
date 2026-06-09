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
from utils.vis_priors import get_unet_and_decoder, register_decoder_taps, agg_norm, save_strip, save_raw_heatmap, \
    shrink_and_gamma, save_priors_overlays
from utils.vis_priors import save_priors_as_individuals  # 顶部有的话可省略
from utils.vis_priors import save_priors_raw_as_individuals  # 新增的函数
import numpy as np
# >>> VIS PRIORS END: imports

# 放在 imports 下面即可
from contextlib import contextmanager
import torch
import re


def _get_batch_name(batch, j=0):
    """尽量从 batch 中取第 j 张样本的原始文件名（不含扩展名）。"""
    def _basename(p):
        b = os.path.basename(str(p))
        return os.path.splitext(b)[0]

    if isinstance(batch, dict):
        # 1) 常见键
        for k in ["img_path", "image_path", "img_paths", "paths", "path",
                  "name", "filename", "fn", "im_name", "file_name"]:
            if k in batch:
                v = batch[k]
                if isinstance(v, (list, tuple)) and len(v) > j:
                    return _basename(v[j])
                if isinstance(v, str):
                    return _basename(v)

        # 2) meta 里套着
        for mk in ["meta", "metas", "img_metas"]:
            if mk in batch:
                m = batch[mk]
                m = m[j] if isinstance(m, (list, tuple)) and len(m) > j else m
                if isinstance(m, dict):
                    for k in ["img_path", "image_path", "path", "name", "filename", "file_name"]:
                        if k in m:
                            return _basename(m[k])

    # 实在没有就返回 None，外面会回退到编号
    return None

def _sanitize_name(s: str) -> str:
    """只保留字母数字._-，避免文件系统不兼容字符。"""
    return re.sub(r"[^0-9a-zA-Z._-]+", "_", s)[:150]

@contextmanager
def cudnn_safe():
    old_e = torch.backends.cudnn.enabled
    old_b = torch.backends.cudnn.benchmark
    old_d = torch.backends.cudnn.deterministic
    try:
        torch.backends.cudnn.enabled = False     # 关键：禁用 cuDNN，走 PyTorch kernel
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        yield
    finally:
        torch.backends.cudnn.enabled = old_e
        torch.backends.cudnn.benchmark = old_b
        torch.backends.cudnn.deterministic = old_d


set_random_seed(7)

# # # 2w预训练!!!
# def get_loader(cfg):
#     # 实例化
#     # casiav1_test_dataset = instantiate_from_config(cfg.test_dataset.CASIAv1)
#     # nist16_test_dataset = instantiate_from_config(cfg.test_dataset.NIST16)
#     columbia_test_dataset = instantiate_from_config(cfg.test_dataset.Columbia)
#     # coverage_test_dataset = instantiate_from_config(cfg.test_dataset.Coverage)
#     # imd20_test_dataset = instantiate_from_config(cfg.test_dataset.IMD20)
#
#     # 创建测试数据加载器
#     # casiav1_test_loader = DataLoader(
#     #     casiav1_test_dataset,
#     #     batch_size=cfg.batch_size,
#     #     collate_fn=collate
#     # )
#     # nist16_test_loader = DataLoader(
#     #     nist16_test_dataset,
#     #     batch_size=cfg.batch_size,
#     #     collate_fn=collate
#     # )
#     columbia_test_loader = DataLoader(
#         columbia_test_dataset,
#         batch_size=cfg.batch_size,
#         collate_fn=collate
#     )
#     # coverage_test_loader = DataLoader(
#     #     coverage_test_dataset,
#     #     batch_size=cfg.batch_size,
#     #     collate_fn=collate
#     # )
#     # imd20_test_loader = DataLoader(
#     #     imd20_test_dataset,
#     #     batch_size=cfg.batch_size,
#     #     collate_fn=collate
#     # )
#
#     return columbia_test_loader
#     # casiav1_test_loader, nist16_test_loader
#     # columbia_test_loader, coverage_test_loader, imd20_test_loader

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

# Inpainting32K
def get_loader(cfg):
    test_dataset = instantiate_from_config(cfg.test_dataset.LR)
    test_dataset = DataLoader(
             test_dataset,
             batch_size=cfg.batch_size,
             collate_fn=collate
         )
    return test_dataset

# CNN-Based-800 GAN-Based-800 TM-Based-800 DM-Based-800
# GC CA SH EC LB RN NS LR PM SG

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # >>> VIS PRIORS BEGIN: cli args
    parser.add_argument("--vis_priors", action="store_true",
                        help="Save prior-fusion heatmaps (Fg/Fl/Fc/c) during inference")
    parser.add_argument("--vis_n", type=int, default=4,
                        help="Number of samples to visualize")
    parser.add_argument("--vis_dir", type=str, default="vis/priors",
                        help="Directory to save visualization strips")
    parser.add_argument('--vis_overlay', action='store_true',
                        help='同时保存叠加原图的可视化')
    parser.add_argument('--vis_overlay_alpha', type=float, default=0.4,
                        help='叠加强度 alpha ∈ [0,1]，越大越“盖住原图”')
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
                        default=['LR'])
    # CASIAv1 Columbia Coverage IMD20 NIST16
    # CASIAv1 Coverage_25 NIST16_160

    # CNN-Based-800 GAN-Based-800 TM-Based-800 DM-Based-800
    # GC CA SH EC LB RN NS LR PM SG

    parser.add_argument('--time_ensemble', action='store_true')
    parser.add_argument('--batch_ensemble', action='store_true')

    cfg = add_args(parser)
    assert not (cfg.time_ensemble and cfg.batch_ensemble), 'Cannot use both time_ensemble and batch_ensemble'
    """
        Hack config here.
    """
    if cfg.num_sample_steps is not None:
        cfg.diffusion_model.params.num_sample_steps = cfg.num_sample_steps

    # CNN-Based-800 GAN-Based-800 TM-Based-800 DM-Based-800
    # GC CA SH EC LB RN NS LR PM SG

    # CASIAv1 Columbia Coverage IMD20 NIST16
    # CASIAv1 Coverage_25 NIST16_160
    LR_test_loader = get_loader(cfg)

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
    # CNN-Based-800 GAN-Based-800 TM-Based-800 DM-Based-800
    # GC CA SH EC LB RN NS LR PM SG

    # CASIAv1 Columbia Coverage IMD20 NIST16
    # CASIAv1 Coverage_25 NIST16_160
    LR_test_loader = trainer.accelerator.prepare(LR_test_loader)

    dataset_map = {
        # 2w  预训练!!!
        # 'CASIAv1': CASIAv1_test_loader,
        # 'NIST16': NIST16_test_loader,
        # 'Coverage': Coverage_test_loader,
        # 'IMD20': IMD20_test_loader,
        # 'Columbia': Columbia_test_loader,

        # 微调
        # 'CASIAv1': CASIAv1_test_loader,
        # 'Coverage_25': Coverage_25_test_loader,
        # 'NIST16_160': NIST16_160_test_loader,

        # Inpainting32K
        'LR': LR_test_loader,
    }
    assert all([d_name in dataset_map.keys() for d_name in cfg.target_dataset]), \
        f'Invalid dataset name. Available dataset: {dataset_map.keys()}' \
        f'Your input: {cfg.target_dataset}'
    target_dataset = [(dataset_map[dataset_name], dataset_name) for dataset_name in cfg.target_dataset]

    for dataset, dataset_name in target_dataset:
        trainer.model.eval()
        # CASIAv1 Columbia Coverage IMD20 NIST16
        # CASIAv1 Coverage_25 NIST16_160
        mask_path = Path(cfg.test_dataset.LR.params.image_root).parent.parent

        save_to = Path(cfg.results_folder) / dataset_name
        os.makedirs(save_to, exist_ok=True)

        # >>> VIS PRIORS BEGIN: visualize a few samples using model.sample (stable path)
        if getattr(cfg, "vis_priors", False) and trainer.accelerator.is_main_process:
            try:
                unet, decoder = get_unet_and_decoder(trainer)
            except Exception as e:
                print(f"[VIS] locate unet/decoder failed: {e}")
                unet, decoder = None, None

            if decoder is not None:
                taps, hook_handles = register_decoder_taps(decoder)
                device = trainer.accelerator.device
                trainer.model.eval()

                vis_dir = getattr(cfg, "vis_dir", "vis/priors")
                os.makedirs(vis_dir, exist_ok=True)


                def _pick_image(batch):
                    # 先尝试常见键名
                    if isinstance(batch, dict):
                        for k in ["image", "img", "I", "cond_img", "input"]:
                            if k in batch and torch.is_tensor(batch[k]) and batch[k].dim() == 4:
                                return batch[k]
                        # 兜底：返回第一个 4D 张量
                        for v in batch.values():
                            if torch.is_tensor(v) and v.dim() == 4:
                                return v
                    elif isinstance(batch, (list, tuple)):
                        for v in batch:
                            if torch.is_tensor(v) and v.dim() == 4:
                                return v
                    raise ValueError("Cannot infer image tensor from batch.")


                vis_count = 0
                for batch in dataset:
                    try:
                        I_b = _pick_image(batch).to(device)  # [B,C,H,W], 已按你的数据预处理
                    except Exception as e:
                        print(f"[VIS] skip batch (no image tensor): {e}")
                        continue

                    # 只取一张，降低显存/算法搜索压力
                    I = I_b[:1]
                    _, _, H, W = I.shape

                    with torch.no_grad():
                        # 为了避免 cuDNN 算法搜不到，临时关掉 benchmark，且不强制 deterministic
                        old_bench = torch.backends.cudnn.benchmark
                        old_det = torch.backends.cudnn.deterministic
                        torch.backends.cudnn.benchmark = False
                        torch.backends.cudnn.deterministic = False
                        try:
                            # 触发一次完整推理（内部会调用 self.model.sample_unet 并带上 extra_cond）
                            with cudnn_safe():  # ← 加这一行
                                pred = trainer.model.sample(I, verbose=False)
                        finally:
                            torch.backends.cudnn.benchmark = old_bench
                            torch.backends.cudnn.deterministic = old_det

                    # 从 hooks 抓到四个特征并规整
                    H_Fg = agg_norm(taps["Fg"], (H, W), reduce="mean")
                    H_Fl = agg_norm(taps["Fl"], (H, W), reduce="mean")
                    H_Fc = agg_norm(taps["Fc"], (H, W), reduce="mean")
                    H_c = agg_norm(taps["c"], (H, W), reduce="mean")

                    H_Fg = shrink_and_gamma(H_Fg, floor=0.20, gamma=1.2, use_percentile=True)  # 背景更深蓝
                    H_Fl = shrink_and_gamma(H_Fl, floor=0.40, gamma=1.2, use_percentile=True)  # 背景更深蓝
                    H_Fc = shrink_and_gamma(H_Fc, floor=0.20, gamma=1.2, use_percentile=True)  # 背景更深蓝
                    H_c = shrink_and_gamma(H_c, floor=0.20, gamma=1.5, use_percentile=True)  # 背景更深蓝

                    # 画图素材
                    img_uint8 = (I[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    pred01 = None
                    if torch.is_tensor(pred):
                        p = pred.detach().cpu().numpy()
                        if p.ndim == 4:
                            p = p[0, 0]
                        pred01 = (p >= 0.5).astype(np.uint8)

                    # save_path = os.path.join(vis_dir, f"{dataset_name}_strip_{vis_count:03d}.png")
                    # save_strip(save_path, img_uint8, H_Fg[0:1], H_Fl[0:1], H_Fc[0:1], H_c[0:1], pred01=pred01)
                    # print(f"[VIS] saved -> {save_path}")
                    # #############################################
                    # 为该样本创建一个小目录（更清晰）：vis/priors/<dataset>_<idx>/

                    # sample_dir = os.path.join(vis_dir, f"{dataset_name}_{vis_count:03d}")
                    # base = f"{dataset_name}_{vis_count:03d}"

                    # j=0 因为你当前每个 batch 只取了第一张 I = I_b[:1]
                    name = _get_batch_name(batch, j=0)
                    if not name:
                        name = f"{dataset_name}_{vis_count:03d}"
                    safe_name = _sanitize_name(name)

                    sample_dir = os.path.join(vis_dir, safe_name)
                    base = safe_name


                    # 逐张保存：Image, Fg, Fl, Fc, c(t), (pred)
                    # save_priors_as_individuals(
                    #     out_dir=sample_dir,
                    #     base=base,
                    #     img_uint8=img_uint8,
                    #     H_Fg=H_Fg[0:1], H_Fl=H_Fl[0:1], H_Fc=H_Fc[0:1], H_c=H_c[0:1],
                    #     pred01=pred01,
                    #     add_colorbar=False  # 需要色条就改 True
                    # )
                    save_priors_raw_as_individuals(
                        out_dir=sample_dir,
                        base=base,
                        H_Fg=H_Fg[0:1], H_Fl=H_Fl[0:1], H_Fc=H_Fc[0:1], H_c=H_c[0:1],
                        cmap="turbo",  # 蓝底高亮
                        add_colorbar=False  # 需要色条就 True
                    )
                    if getattr(cfg, "vis_overlay", False):
                        save_priors_overlays(
                            out_dir=sample_dir, base=base,
                            img_uint8=img_uint8,
                            H_Fg=H_Fg[0:1], H_Fl=H_Fl[0:1], H_Fc=H_Fc[0:1], H_c=H_c[0:1],
                            alpha=float(getattr(cfg, "vis_overlay_alpha", 0.4)),
                            cmap="turbo",  # 保持和 raw 一致；想换别的如 "jet" 也可
                            add_colorbar=False
                        )

                    print(f"[VIS] saved -> {sample_dir} (Image/Fg/Fl/Fc/c(t){'/pred' if pred01 is not None else ''})")
                    # ###################################

                    vis_count += 1
                    if vis_count >= getattr(cfg, "vis_n", 4):
                        break

                # 记得移除 hooks（避免后续评测重复挂载）
                for _h in hook_handles:
                    _h.remove()
        # >>> VIS PRIORS END


        if cfg.batch_ensemble:
            # mae-->auc
            auc, _ = trainer.val_batch_ensemble(model=trainer.model,
                                                test_data_loader=dataset,
                                                accelerator=trainer.accelerator,
                                                thresholding=False,
                                                save_to=save_to)
        elif cfg.time_ensemble:
            with torch.no_grad():
                with cudnn_safe():
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
