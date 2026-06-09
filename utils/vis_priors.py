# utils/vis_priors.py
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---------- [B,C,h,w] -> [B,1,H,W]，通道聚合 + 上采样 + min-max 归一化 ----------
def agg_norm(feat: torch.Tensor, out_hw, reduce="mean"):
    if feat.dim() == 3:
        feat = feat.unsqueeze(0)
    if reduce == "mean":
        Hmap = feat.mean(1, keepdim=True)
    elif reduce == "max":
        Hmap = feat.amax(1, keepdim=True)
    else:
        raise ValueError("reduce must be 'mean' or 'max'")
    if Hmap.shape[-2:] != out_hw:
        Hmap = F.interpolate(Hmap, size=out_hw, mode="bilinear", align_corners=False)
    mn = Hmap.amin(dim=(-2, -1), keepdim=True)
    mx = Hmap.amax(dim=(-2, -1), keepdim=True)
    Hmap = (Hmap - mn) / (mx - mn + 1e-8)
    return Hmap

# ---------- 横向条带：Image | Fg | Fl | Fc | c(t) | (可选 prediction) ----------
def save_strip(save_path, img_uint8, H_Fg, H_Fl, H_Fc, H_c, pred01=None, alpha=0.5, cmap="magma"):
    cols = 6 if pred01 is None else 7
    titles = ["Image", r"$F_g$", r"$F_l$", r"$F_c$", r"$c(t)$"] + (["prediction"] if pred01 is not None else [])
    maps = [None, H_Fg, H_Fl, H_Fc, H_c] + ([pred01] if pred01 is not None else [])

    import matplotlib
    matplotlib.use("Agg")  # 保险起见：无显示环境也能保存

    fig, axes = plt.subplots(1, cols, figsize=(14, 2.2))
    axes[0].imshow(img_uint8); axes[0].set_title(titles[0]); axes[0].axis("off")
    for i, M in enumerate(maps[1:], start=1):
        ax = axes[i]
        ax.imshow(img_uint8)
        if isinstance(M, torch.Tensor):
            ax.imshow(M.squeeze().detach().cpu(), alpha=alpha, cmap=cmap, vmin=0, vmax=1)
        else:
            pm = np.zeros_like(img_uint8, dtype=np.uint8)  # 红色掩码覆盖
            pm[..., 0] = (M * 255).astype(np.uint8)
            ax.imshow(pm, alpha=alpha)
        ax.set_title(titles[i]); ax.axis("off")
    plt.tight_layout(pad=0.1)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

# ---------- 在 Decoder 上注册 4 个 forward hook ----------
def register_decoder_taps(decoder):
    """
    对应 net.py::Decoder：
      Fc:  linear_fuse1(...)
      Fg:  conv_block(...)
      Fl:  bdlu(...)[0]   # tuple 的第一个
      c :  Tri_TFF(...)
    """
    taps = {}
    def h_fc(module, inputs, output):   # _c
        taps["Fc"] = output
    def h_xf(module, inputs, output):   # xf
        taps["Fg"] = output
    def h_bdlu(module, inputs, output): # (xl, pred_edge)
        if isinstance(output, tuple):
            xl, pred_edge = output
            taps["xl"] = xl  # 明确保留 xl
            taps["edge"] = pred_edge  # 单通道边缘图
            taps["Fl"] = pred_edge  # << 如果你希望 Fl 就是“边缘先验”，用这行
        else:
            taps["Fl"] = output
    def h_c(module, inputs, output):    # TriXFusion
        taps["c"] = output

    h1 = decoder.linear_fuse1.register_forward_hook(h_fc)
    h2 = decoder.conv_block.register_forward_hook(h_xf)
    h3 = decoder.bdlu.register_forward_hook(h_bdlu)
    h4 = decoder.Tri_TFF.register_forward_hook(h_c)
    return taps, [h1, h2, h3, h4]

# ---------- 从 Trainer 里稳健定位 unet 与 decode_head ----------
def get_unet_and_decoder(trainer):
    """
    适配你的构建链：diffusion_model(model=...), model 里持有 net，并暴露 decode_head
    """
    mdl = getattr(trainer, "model", trainer)
    candidates_unet = [
        "model.unet",        # 常见：diffusion_model.model.unet
        "unet",              # 直接挂在最外层
        "model",             # 如果最外层就是 net
    ]
    candidates_dec = [
        "model.unet.decode_head",
        "unet.decode_head",
        "model.decode_head",
        "decode_head",
    ]

    def _resolve(root, dotted):
        obj = root
        for name in dotted.split("."):
            if not hasattr(obj, name): return None
            obj = getattr(obj, name)
        return obj

    unet = None
    for p in candidates_unet:
        obj = _resolve(mdl, p)
        if obj is not None:
            unet = obj; break
    if unet is None:
        raise AttributeError("Cannot locate unet from trainer.model; please adjust get_unet_and_decoder().")

    decoder = None
    for p in candidates_dec:
        obj = _resolve(mdl, p)
        if obj is not None:
            decoder = obj; break
    if decoder is None:
        raise AttributeError("Cannot locate decode_head (Decoder) from trainer.model; please adjust paths.")

    return unet, decoder

# === 保存「单张」覆盖图：底图+热力图 ===
def save_overlay(save_path, img_uint8, heat_tensor, alpha=0.5, cmap="magma", add_colorbar=False, title=None):
    """
    heat_tensor: [1,1,H,W] in [0,1] (已归一化)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H = heat_tensor.squeeze().detach().cpu().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2))
    ax.imshow(img_uint8)
    im = ax.imshow(H, alpha=alpha, cmap=cmap, vmin=0, vmax=1)
    if title is not None:
        ax.set_title(title)
    ax.axis("off")
    if add_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

# === 保存「单张」预测掩码覆盖（红色半透明）===
def save_mask_overlay(save_path, img_uint8, pred01, alpha=0.35, title="prediction"):
    """
    pred01: [H,W] 0/1 numpy
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2))
    ax.imshow(img_uint8)
    pm = np.zeros_like(img_uint8, dtype=np.uint8)
    pm[..., 0] = (pred01 * 255).astype(np.uint8)  # 红色通道
    ax.imshow(pm, alpha=alpha)
    if title is not None:
        ax.set_title(title)
    ax.axis("off")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

# === 批量辅助：按样本建立目录并逐张保存 ===
def save_priors_as_individuals(out_dir, base, img_uint8, H_Fg, H_Fl, H_Fc, H_c, pred01=None, add_colorbar=False):
    """
    H_*: [1,1,H,W] torch tensor in [0,1]
    """
    os.makedirs(out_dir, exist_ok=True)
    save_overlay(os.path.join(out_dir, f"{base}_image.png"), img_uint8, H_c*0, alpha=0.0, title=None)  # 原图
    save_overlay(os.path.join(out_dir, f"{base}_Fg.png"),    img_uint8, H_Fg, add_colorbar=add_colorbar, title=None)
    save_overlay(os.path.join(out_dir, f"{base}_Fl.png"),    img_uint8, H_Fl, add_colorbar=add_colorbar, title=None)
    save_overlay(os.path.join(out_dir, f"{base}_Fc.png"),    img_uint8, H_Fc, add_colorbar=add_colorbar, title=None)
    save_overlay(os.path.join(out_dir, f"{base}_c_t.png"),   img_uint8, H_c,  add_colorbar=add_colorbar, title=None)
    if pred01 is not None:
        save_mask_overlay(os.path.join(out_dir, f"{base}_pred.png"), img_uint8, pred01, title=None)

# === 仅保存纯热力图（不叠原图）；默认蓝底高亮 ===
def save_raw_heatmap(save_path, heat_tensor, cmap="jet", add_colorbar=False):
    """
    heat_tensor: [1,1,H,W] in [0,1]（已归一化）
    cmap: "Blues_r" 表示低值=深蓝、 高值=亮（白），即“蓝底高亮”
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    H = heat_tensor.squeeze().detach().cpu().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2))
    im = ax.imshow(H, cmap=cmap, vmin=0, vmax=1)
    ax.axis("off")
    if add_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

# === 批量：分别保存 Fg/Fl/Fc/c(t) 的纯热力图 ===
def save_priors_raw_as_individuals(out_dir, base, H_Fg, H_Fl, H_Fc, H_c,
                                   cmap="turbo", add_colorbar=False):
    """
    H_*: [1,1,H,W] torch tensor in [0,1]
    """
    os.makedirs(out_dir, exist_ok=True)
    save_raw_heatmap(os.path.join(out_dir, f"{base}_Fg.png"),  H_Fg, cmap=cmap, add_colorbar=add_colorbar)
    save_raw_heatmap(os.path.join(out_dir, f"{base}_Fl.png"),  H_Fl, cmap=cmap, add_colorbar=add_colorbar)
    save_raw_heatmap(os.path.join(out_dir, f"{base}_Fc.png"),  H_Fc, cmap=cmap, add_colorbar=add_colorbar)
    save_raw_heatmap(os.path.join(out_dir, f"{base}_c.png"), H_c,  cmap=cmap, add_colorbar=add_colorbar)

# 把低响应压到更接近 0（深蓝），再用 gamma 提升高亮对比
def shrink_and_gamma(H, floor=0.15, gamma=1.3, use_percentile=False):
    """
    H: [B,1,H,W] in [0,1]
    floor: 绝对地板(0~1)。比如 0.15 表示把 <=0.15 的响应压成 0。
    use_percentile=True 时，floor 被当成分位数（如 0.15→第15百分位）。
    """
    if use_percentile:
        try:
            q = torch.quantile(H, floor, dim=(-2, -1), keepdim=True)
        except Exception:
            q = H.flatten(-2).kthvalue(int(max(1, floor * H.shape[-2] * H.shape[-1])) , dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        H = (H - q).clamp_min(0) / (1 - q + 1e-8)
    else:
        H = (H - floor).clamp_min(0) / (1 - floor + 1e-8)
    if gamma != 1.0:
        H = H.clamp(0, 1) ** gamma
    return H.clamp(0, 1)

# === 把热力图叠加到原图并保存 ===
def save_overlay(save_path, img_uint8, heat_tensor, alpha=0.4, cmap="turbo", add_colorbar=False):
    """
    img_uint8: [H,W,3] uint8 原图
    heat_tensor: [1,1,H,W] in [0,1] 已归一化的热力图
    alpha: 叠加强度（0~1）
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H = heat_tensor.squeeze().detach().cpu().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2))
    ax.imshow(img_uint8)
    im = ax.imshow(H, alpha=float(alpha), cmap=cmap, vmin=0, vmax=1)
    ax.axis("off")
    if add_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

# === 批量：分别保存 Fg/Fl/Fc/c(t) 的叠加图 ===
def save_priors_overlays(out_dir, base, img_uint8, H_Fg, H_Fl, H_Fc, H_c,
                         alpha=0.4, cmap="turbo", add_colorbar=False):
    os.makedirs(out_dir, exist_ok=True)
    save_overlay(os.path.join(out_dir, f"{base}_Fg_ovl.png"),  img_uint8, H_Fg, alpha=alpha, cmap=cmap, add_colorbar=add_colorbar)
    save_overlay(os.path.join(out_dir, f"{base}_Fl_ovl.png"),  img_uint8, H_Fl, alpha=alpha, cmap=cmap, add_colorbar=add_colorbar)
    save_overlay(os.path.join(out_dir, f"{base}_Fc_ovl.png"),  img_uint8, H_Fc, alpha=alpha, cmap=cmap, add_colorbar=add_colorbar)
    save_overlay(os.path.join(out_dir, f"{base}_c_t_ovl.png"), img_uint8, H_c,  alpha=alpha, cmap=cmap, add_colorbar=add_colorbar)
