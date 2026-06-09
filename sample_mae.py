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

set_random_seed(7)

# 2w预训练!!!
# def get_loader(cfg):
#     # 实例化
#     # casiav1_test_dataset = instantiate_from_config(cfg.test_dataset.CASIAv1)
#     nist16_test_dataset = instantiate_from_config(cfg.test_dataset.NIST16)
#     # columbia_test_dataset = instantiate_from_config(cfg.test_dataset.Columbia)
#     # coverage_test_dataset = instantiate_from_config(cfg.test_dataset.Coverage)
#     # imd20_test_dataset = instantiate_from_config(cfg.test_dataset.IMD20)
#
#     # 创建测试数据加载器
#     # casiav1_test_loader = DataLoader(
#     #     casiav1_test_dataset,
#     #     batch_size=cfg.batch_size,
#     #     collate_fn=collate
#     # )
#     nist16_test_loader = DataLoader(
#         nist16_test_dataset,
#         batch_size=cfg.batch_size,
#         collate_fn=collate
#     )
#     # columbia_test_loader = DataLoader(
#     #     columbia_test_dataset,
#     #     batch_size=cfg.batch_size,
#     #     collate_fn=collate
#     # )
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
#     return nist16_test_loader
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

# # fine_Cover
# def get_loader(cfg):
#     Cover_25_test_dataset = instantiate_from_config(cfg.test_dataset.Coverage_25)
#     Cover_25_test_dataset = DataLoader(
#              Cover_25_test_dataset,
#              batch_size=cfg.batch_size,
#              collate_fn=collate
#          )
#     return Cover_25_test_dataset

# fine_NIST
def get_loader(cfg):
    NIST_160_test_dataset = instantiate_from_config(cfg.test_dataset.NIST16_160)
    NIST_160_test_dataset = DataLoader(
             NIST_160_test_dataset,
             batch_size=cfg.batch_size,
             collate_fn=collate
         )
    return NIST_160_test_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--results_folder', type=str, default='./results')
    parser.add_argument('--num_epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulate_every', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_sample_steps', type=int, default=None)

    parser.add_argument('--target_dataset', nargs='+', type=str,
                        default=['NIST16_160'])
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
    NIST16_160_test_loader = get_loader(cfg)

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
    NIST16_160_test_loader = trainer.accelerator.prepare(NIST16_160_test_loader)

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
        'NIST16_160': NIST16_160_test_loader,
    }
    assert all([d_name in dataset_map.keys() for d_name in cfg.target_dataset]), \
        f'Invalid dataset name. Available dataset: {dataset_map.keys()}' \
        f'Your input: {cfg.target_dataset}'
    target_dataset = [(dataset_map[dataset_name], dataset_name) for dataset_name in cfg.target_dataset]

    for dataset, dataset_name in target_dataset:
        trainer.model.eval()
        # CASIAv1 Columbia Coverage IMD20 NIST16
        # CASIAv1 Coverage_25 NIST16_160
        mask_path = Path(cfg.test_dataset.NIST16_160.params.image_root).parent.parent

        save_to = Path(cfg.results_folder) / dataset_name
        os.makedirs(save_to, exist_ok=True)
        if cfg.batch_ensemble:
            # mae-->auc
            mae, _ = trainer.val_batch_ensemble(model=trainer.model,
                                                test_data_loader=dataset,
                                                accelerator=trainer.accelerator,
                                                thresholding=False,
                                                save_to=save_to)
        elif cfg.time_ensemble:
            mae, _ = trainer.val_time_ensemble(model=trainer.model,
                                               test_data_loader=dataset,
                                               accelerator=trainer.accelerator,
                                               thresholding=False,
                                               save_to=save_to)
        else:
            mae, _ = trainer.val(model=trainer.model,
                                 test_data_loader=dataset,
                                 accelerator=trainer.accelerator,
                                 thresholding=False,
                                 save_to=save_to)
        trainer.accelerator.wait_for_everyone()
        trainer.accelerator.print(f'{dataset_name} mae: {mae}')

        if trainer.accelerator.is_main_process:
            from utils.eval import eval

            eval_score = eval(
                mask_path=mask_path,
                pred_path=cfg.results_folder,
                dataset_name=dataset_name)
        trainer.accelerator.wait_for_everyone()
