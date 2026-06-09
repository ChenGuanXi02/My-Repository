import os
import time

from utils import init_env

import argparse

import torch
from utils.collate_utils import collate, SampleDataset
from utils.import_utils import instantiate_from_config, recurse_instantiate_from_config, get_obj_from_str
from utils.init_utils import add_args, config_pretty
from utils.train_utils import set_random_seed
from torch.utils.data import DataLoader
from utils.trainer import Trainer

set_random_seed(42)

# # # # # #预训练2w# # # # #
# def get_loader(cfg):
#     train_dataset = instantiate_from_config(cfg.train_dataset)
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=cfg.batch_size,
#         shuffle=True,
#         num_workers=cfg.num_workers)
#     # # old 泄露
#     # test_dataset = instantiate_from_config(cfg.test_dataset.Coverage)
#     # test_dataset_expand = SampleDataset(full_dataset=instantiate_from_config(cfg.test_dataset.CASIAv1), interval=10)
#     # test_dataset = torch.utils.data.ConcatDataset([test_dataset, test_dataset_expand])
#     # test_dataset_expand = SampleDataset(full_dataset=instantiate_from_config(cfg.test_dataset.Columbia), interval=2)
#     # test_dataset = torch.utils.data.ConcatDataset([test_dataset, test_dataset_expand])
#     # test_dataset_expand = SampleDataset(full_dataset=instantiate_from_config(cfg.test_dataset.IMD20), interval=20)
#     # test_dataset = torch.utils.data.ConcatDataset([test_dataset, test_dataset_expand])
#     # test_dataset_expand = SampleDataset(full_dataset=instantiate_from_config(cfg.test_dataset.NIST16), interval=5)
#     # test_dataset = torch.utils.data.ConcatDataset([test_dataset, test_dataset_expand])
#
#     # # 2w(HDF-Net 80% train 20% val)
#     test_dataset = instantiate_from_config(cfg.test_dataset.HDFNet)
#     # test_dataset = instantiate_from_config(cfg.test_dataset.CASIAv1)
#
#     # # Ceshi
#     # test_dataset = instantiate_from_config(cfg.test_dataset.CeShi)
#
#     # 创建测试数据加载器
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=cfg.batch_size,
#         collate_fn=collate
#     )
#     return train_loader, test_loader

# # # #微调Casia
# def get_loader(cfg):
#     train_dataset = instantiate_from_config(cfg.train_dataset)
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=cfg.batch_size,
#         shuffle=True,
#         num_workers=cfg.num_workers)
#     test_dataset_casiav1 = instantiate_from_config(cfg.test_dataset.CASIAv1)
#     test_dataset = torch.utils.data.ConcatDataset([test_dataset_casiav1])
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=cfg.batch_size,
#         collate_fn=collate
#     )
#     return train_loader, test_loader

# # 微调Cover
# def get_loader(cfg):
#     train_dataset = instantiate_from_config(cfg.train_dataset)
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=cfg.batch_size,
#         shuffle=True,
#         num_workers=cfg.num_workers)
#     test_dataset_coverage_25 = instantiate_from_config(cfg.test_dataset.Coverage_25)
#     test_dataset = torch.utils.data.ConcatDataset([test_dataset_coverage_25])
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=cfg.batch_size,
#         collate_fn=collate
#     )
#     return train_loader, test_loader
#
# # 微调Nist
# def get_loader(cfg):
#     train_dataset = instantiate_from_config(cfg.train_dataset)
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=cfg.batch_size,
#         shuffle=True,
#         num_workers=cfg.num_workers)
#     test_dataset_nist_160 = instantiate_from_config(cfg.test_dataset.NIST16_160)
#     test_dataset = torch.utils.data.ConcatDataset([test_dataset_nist_160])
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=cfg.batch_size,
#         collate_fn=collate
#     )
#     return train_loader, test_loader

# # # casiav2 Train Defacto6k Sample
# def get_loader(cfg):
#     train_dataset = instantiate_from_config(cfg.train_dataset)
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=cfg.batch_size,
#         shuffle=True,
#         num_workers=cfg.num_workers)
#
#     # defacto6k
#     test_dataset = instantiate_from_config(cfg.test_dataset.DefactoCopyMove)
#     test_dataset_2 = instantiate_from_config(cfg.test_dataset.DefactoSplicing)
#     test_dataset_3 = instantiate_from_config(cfg.test_dataset.DefactoInpainting)
#     test_dataset = torch.utils.data.ConcatDataset([test_dataset, test_dataset_2, test_dataset_3])
#
#     # 创建测试数据加载器
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=cfg.batch_size,
#         collate_fn=collate
#     )
#     return train_loader, test_loader

# # DiffForencis训练法
# def get_loader(cfg):
#     train_dataset = instantiate_from_config(cfg.train_dataset)
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=cfg.batch_size,
#         shuffle=True,
#         num_workers=cfg.num_workers)
#
#     test_dataset = instantiate_from_config(cfg.test_dataset.CASIAv1)
#
#     # 创建测试数据加载器
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=cfg.batch_size,
#         collate_fn=collate
#     )
#     return train_loader, test_loader

# Inpainting32K训练法
def get_loader(cfg):
    train_dataset = instantiate_from_config(cfg.train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers)

    test_dataset = instantiate_from_config(cfg.test_dataset.GC)
    # test_dataset_expand = SampleDataset(full_dataset=instantiate_from_config(cfg.test_dataset.CA), interval=1)
    # test_dataset = torch.utils.data.ConcatDataset([test_dataset, test_dataset_expand])
    test_dataset_expand = SampleDataset(full_dataset=instantiate_from_config(cfg.test_dataset.SH), interval=1)
    test_dataset = torch.utils.data.ConcatDataset([test_dataset, test_dataset_expand])
    test_dataset_expand = SampleDataset(full_dataset=instantiate_from_config(cfg.test_dataset.EC), interval=1)
    test_dataset = torch.utils.data.ConcatDataset([test_dataset, test_dataset_expand])
    test_dataset_expand = SampleDataset(full_dataset=instantiate_from_config(cfg.test_dataset.LB), interval=1)
    test_dataset = torch.utils.data.ConcatDataset([test_dataset, test_dataset_expand])
    test_dataset_expand = SampleDataset(full_dataset=instantiate_from_config(cfg.test_dataset.RN), interval=1)
    test_dataset = torch.utils.data.ConcatDataset([test_dataset, test_dataset_expand])
    # test_dataset_expand = SampleDataset(full_dataset=instantiate_from_config(cfg.test_dataset.NS), interval=1)
    # test_dataset = torch.utils.data.ConcatDataset([test_dataset, test_dataset_expand])
    test_dataset_expand = SampleDataset(full_dataset=instantiate_from_config(cfg.test_dataset.LR), interval=1)
    test_dataset = torch.utils.data.ConcatDataset([test_dataset, test_dataset_expand])
    test_dataset_expand = SampleDataset(full_dataset=instantiate_from_config(cfg.test_dataset.PM_Old), interval=1)
    test_dataset = torch.utils.data.ConcatDataset([test_dataset, test_dataset_expand])
    test_dataset_expand = SampleDataset(full_dataset=instantiate_from_config(cfg.test_dataset.SG), interval=1)
    test_dataset = torch.utils.data.ConcatDataset([test_dataset, test_dataset_expand])


    # 创建测试数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate
    )
    return train_loader, test_loader


# 乱成一锅粥的原版
# def get_loader(cfg):
#     train_dataset = instantiate_from_config(cfg.train_dataset)
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=cfg.batch_size,
#         shuffle=True,
#         num_workers=cfg.num_workers)
#     # 预训练2w启用tttttttttttttttttttttttttt：
#     # 实例化新的数据集
#     test_dataset_casia = instantiate_from_config(cfg.test_dataset.CASIAv1)
#     # test_dataset_nist = instantiate_from_config(cfg.test_dataset.NIST16)
#     # test_dataset_columbia = instantiate_from_config(cfg.test_dataset.Columbia)
#     # test_dataset_coverage = instantiate_from_config(cfg.test_dataset.Coverage)
#     # test_dataset_imd20 = instantiate_from_config(cfg.test_dataset.IMD20)
#
#     # fffffffffffffffffffffffffffffffffffff
#     # 微调启用
#     # 实例化新的数据集
#     # test_dataset_casia = instantiate_from_config(cfg.test_dataset.CASIAv1)
#     # test_dataset_nist_160 = instantiate_from_config(cfg.test_dataset.NIST16_160)
#     # test_dataset_columbia = instantiate_from_config(cfg.test_dataset.Columbia)
#     # test_dataset_coverage_25 = instantiate_from_config(cfg.test_dataset.Coverage_25)
#     # test_dataset_imd20 = instantiate_from_config(cfg.test_dataset.IMD20)
#
#     # 直接合并数据集，不使用采样间隔
#     # v2
#     # test_dataset = torch.utils.data.ConcatDataset([test_dataset_casia, test_dataset_nist,
#     #                                                test_dataset_columbia, test_dataset_coverage,
#     #                                                test_dataset_imd20])
#     test_dataset = torch.utils.data.ConcatDataset([test_dataset_casia])
#
#     # v3
#     # test_dataset = torch.utils.data.ConcatDataset([test_dataset_nist_160])
#     # [test_dataset_casia, test_dataset_nist_160,
#     #  test_dataset_columbia, test_dataset_coverage_25,
#     #  test_dataset_imd20]
#     # ############################################################################
#     # 创建测试数据加载器
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=cfg.batch_size,
#         collate_fn=collate
#     )
#     return train_loader, test_loader
#
#     # row
#     # test_dataset = instantiate_from_config(cfg.test_dataset.CAMO)
#     # test_dataset_expand = SampleDataset(full_dataset=instantiate_from_config(cfg.test_dataset.COD10K), interval=10)
#     # test_dataset = torch.utils.data.ConcatDataset([test_dataset, test_dataset_expand])
#     # test_dataset_expand = SampleDataset(full_dataset=instantiate_from_config(cfg.test_dataset.NC4K), interval=30)
#     # test_dataset = torch.utils.data.ConcatDataset([test_dataset, test_dataset_expand])
#     #
#     # test_loader = DataLoader(
#     #     test_dataset,
#     #     batch_size=cfg.batch_size,
#     #     collate_fn=collate
#     # )
#     # return train_loader, test_loader
#
#     # # v1
#     # # # 获取测试数据集
#     # test_dataset = instantiate_from_config(cfg.test_dataset.COD10K)
#     # test_loader = DataLoader(
#     #     test_dataset,
#     #     batch_size=cfg.batch_size,
#     #     collate_fn=collate
#     # )
#     # return train_loader, test_loader



if __name__ == "__main__":

    # os.environ["WANDB_API_KEY"] = '1b80dff559a235a9924df6e8377023e85b66af54'
    # os.environ["WANDB_MODE"] = "offline"

    # os.environ["WANDB_MODE"] = "disabled"

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--fp16', action='store_true')
    # parser.add_argument('--results_folder', type=str, default=None, help='None for saving in wandb folder.')
    parser.add_argument('--results_folder', type=str, default='./results', help='本地保存目录')

    parser.add_argument('--num_epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulate_every', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr_min', type=float, default=1e-6)

    cfg = add_args(parser)

    config_pretty(cfg)

    # 新增：生成时间戳（格式：yyMMdd_HHmm，如250531_2255）
    current_time = time.strftime("%y%m%d_%H%M")
    # 构造新的results路径（原路径+时间戳子目录）
    new_results_folder = os.path.join(cfg.results_folder, current_time)
    # 更新配置中的results路径
    cfg.results_folder = new_results_folder
    # 确保新目录存在（若不存在则创建）
    os.makedirs(cfg.results_folder, exist_ok=True)


    cond_uvit = instantiate_from_config(cfg.cond_uvit,
                                        conditioning_klass=get_obj_from_str(cfg.cond_uvit.params.conditioning_klass))
    model = recurse_instantiate_from_config(cfg.model,
                                            unet=cond_uvit)
    diffusion_model = instantiate_from_config(cfg.diffusion_model,
                                              model=model)


    train_loader, test_loader = get_loader(cfg)

    optimizer = instantiate_from_config(cfg.optimizer, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epoch, eta_min=cfg.lr_min)


    trainer = Trainer(
        diffusion_model, train_loader, test_loader,
        train_val_forward_fn=get_obj_from_str(cfg.train_val_forward_fn),
        gradient_accumulate_every=cfg.gradient_accumulate_every,
        results_folder=cfg.results_folder,
        optimizer=optimizer, scheduler=scheduler,
        train_num_epoch=cfg.num_epoch,
        amp=cfg.fp16,
        # log_with=None if cfg.num_workers == 0 else 'wandb',  # debug
        cfg=cfg,
    )
    if getattr(cfg, 'resume', None) or getattr(cfg, 'pretrained', None):
        trainer.load(resume_path=cfg.resume, pretrained_path=cfg.pretrained)
    trainer.train()
