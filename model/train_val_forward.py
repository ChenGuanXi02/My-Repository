import torch
from torch import nn
import torch.nn.functional as F



def normalize_to_01(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def simple_train_val_forward(model: nn.Module, gt=None, image=None, **kwargs):
    if model.training:
        assert gt is not None and image is not None
        out = model(gt, image, **kwargs)
        print("看我看我在 train_val_forward.py 的simple_train_val_forward的model output:", out)
        return out  # 原输出可能只有 pred 或包含其他信息
        # print("在 train_val_forward.py 的simple_train_val_forward的model output:", model(gt, image, seg=seg, **kwargs))
        # pred, pred_edge = model(gt, image, **kwargs)  # 模型返回主预测和边缘预测
        # return {
        #     "pred": pred,  # 主分割预测
        #     "pred_edge": pred_edge,  # 边缘预测
        #     "image": image,
        #     "gt": gt
        # }
    else:
        time_ensemble = kwargs.pop('time_ensemble') if 'time_ensemble' in kwargs else False
        gt_sizes = kwargs.pop('gt_sizes') if time_ensemble else None
        pred = model.sample(image, **kwargs)
        if time_ensemble:
            preds = torch.concat(model.history, dim=1).detach().cpu()
            pred = torch.mean(preds, dim=1, keepdim=True)

            def process(i, p, gt_size):
                p = F.interpolate(p.unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                p = normalize_to_01(p)
                ps = F.interpolate(preds[i].unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                preds_round = (ps > 0).float().mean(dim=1, keepdim=True)
                p_postion = (preds_round > 0.5).float()
                p = p_postion * p
                return p

            pred = [process(index, p, gt_size) for index, (p, gt_size) in enumerate(zip(pred, gt_sizes))]
        return {
            "image": image,
            "pred": pred,
            "gt": gt if gt is not None else None,
        }


def modification_train_val_forward(model: nn.Module, gt=None, image=None, seg=None, **kwargs):
    """This is for the modification task. When diffusion model add noise, will use seg instead of gt."""
    if model.training:
        assert gt is not None and image is not None and seg is not None
        return model(gt, image, seg=seg, **kwargs)
        # return model(gt, image, **kwargs)  # 原输出可能只有 pred 或包含其他信息
        # print("在 train_val_forward.py 的modification_train_val_forward的model output:", model(gt, image, seg=seg, **kwargs))
        # pred, pred_edge = model(gt, image, seg=seg, **kwargs)  # 模型返回主预测和边缘预测
        # return {
        #     "pred": pred,  # 主分割预测
        #     "pred_edge": pred_edge,  # 边缘预测
        #     "image": image,
        #     "gt": gt,
        #     "seg": seg
        # }
    else:
        time_ensemble = kwargs.pop('time_ensemble') if 'time_ensemble' in kwargs else False
        gt_sizes = kwargs.pop('gt_sizes') if time_ensemble else None
        pred = model.sample(image, **kwargs).detach().cpu()
        if time_ensemble:
            """ Here is the function 3, Uncertainty based"""
            preds = torch.concat(model.history, dim=1).detach().cpu()
            pred = torch.mean(preds, dim=1, keepdim=True)

            def process(i, p, gt_size):
                p = F.interpolate(p.unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                p = normalize_to_01(p)
                ps = F.interpolate(preds[i].unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                preds_round = (ps > 0).float().mean(dim=1, keepdim=True)
                p_postion = (preds_round > 0.5).float()
                p = p_postion * p
                return p

            pred = [process(index, p, gt_size) for index, (p, gt_size) in enumerate(zip(pred, gt_sizes))]
        return {
            "image": image,
            "pred": pred,
            "gt": gt if gt is not None else None,
        }


def modification_train_val_forward_e(model: nn.Module, gt=None, image=None, seg=None, **kwargs):
    """This is for the modification task. When diffusion model add noise, will use seg instead of gt."""
    if model.training:
        assert gt is not None and image is not None and seg is not None
        return model(gt, image, seg=seg, **kwargs)
    else:
        time_ensemble = kwargs.pop('time_ensemble') if 'time_ensemble' in kwargs else False
        gt_sizes = kwargs.pop('gt_sizes') if time_ensemble else None
        pred = model.sample(image, **kwargs).detach().cpu()
        if time_ensemble:
            """ Here is extend function 4, with batch extend."""
            preds = torch.concat(model.history, dim=1).detach().cpu()
            for i in range(2):
                model.sample(image, **kwargs)
                preds = torch.cat([preds, torch.concat(model.history, dim=1).detach().cpu()], dim=1)
            pred = torch.mean(preds, dim=1, keepdim=True)

            def process(i, p, gt_size):
                p = F.interpolate(p.unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                p = normalize_to_01(p)
                ps = F.interpolate(preds[i].unsqueeze(0), size=gt_size, mode='bilinear', align_corners=False)
                preds_round = (ps > 0).float().mean(dim=1, keepdim=True)
                p_postion = (preds_round > 0.5).float()
                p = p_postion * p
                return p

            pred = [process(index, p, gt_size) for index, (p, gt_size) in enumerate(zip(pred, gt_sizes))]
        return {
            "image": image,
            "pred": pred,
            "gt": gt if gt is not None else None,
        }
