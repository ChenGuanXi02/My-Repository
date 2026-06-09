import glob
import os
from collections import defaultdict
from pathlib import Path

import math
import numpy as np
import torch
from tqdm import tqdm
import wandb
from accelerate import Accelerator

from model.loss import structure_loss
# from utils.logger_utils import create_url_shortcut_of_wandb, create_logger_of_wandb, create_logger
from utils.logger_utils import create_logger
from utils.train_utils import SmoothedValue, set_random_seed
from utils.import_utils import fill_args_from_dict
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model.train_val_forward import simple_train_val_forward

from sklearn.metrics import roc_auc_score

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def exists(x):
    return x is not None


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def cal_mae(gt, res, thresholding, save_to=None, n=None):
    res = F.interpolate(res.unsqueeze(0), size=gt.shape, mode='bilinear', align_corners=False)
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    res = (res > 0.5).float() if thresholding else res
    res = res.cpu().numpy().squeeze()
    if save_to is not None:
        # plt.imsave(os.path.join(save_to, n), res, cmap='gray')
        # print('111111')
        # print(save_to)
        # print(n)
        # print(os.path.join(save_to, n))
        plt.imsave(os.path.join(save_to, n), res, cmap='gray', format = 'png')
    return np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

def cal_auc(gt: np.ndarray, pred: np.ndarray) -> float:
    pred_prob = pred / 255.0 if pred.max() > 1 else pred  # 归一化
    gt_labels = (gt > 0).astype(int)

    # print(f"[DEBUG] pred min: {pred.min()}, max: {pred.max()}, gt unique: {np.unique(gt_labels)}")

    try:
        auc_score = roc_auc_score(gt_labels.flatten(), pred_prob.flatten())
    except ValueError:
        auc_score = 0.0
    return auc_score


def run_on_seed(func):
    def wrapper(*args, **kwargs):
        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        set_random_seed(0)
        res = func(*args, **kwargs)
        set_random_seed(seed)
        return res

    return wrapper


class Trainer(object):
    def __init__(
            self,
            model,
            train_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader = None,
            train_val_forward_fn=simple_train_val_forward,
            gradient_accumulate_every=1,
            optimizer=None, scheduler=None,
            train_num_epoch=100,
            results_folder='./results',
            amp=False,
            fp16=False,
            split_batches=True,
            # log_with='wandb',
            cfg=None,
    ):
        super().__init__()
        """
            Initialize the accelerator.
        """
        from accelerate import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no',
            # log_with='wandb' if log_with else None,
            gradient_accumulation_steps=gradient_accumulate_every,
            kwargs_handlers=[ddp_kwargs]
        )
        project_name = getattr(cfg, "project_name", 'ResidualDiffsuion-v7')
        self.accelerator.init_trackers(project_name, config=cfg)
        # create_url_shortcut_of_wandb(accelerator=self.accelerator)
        # self.logger = create_logger_of_wandb(accelerator=self.accelerator, rank=not self.accelerator.is_main_process)
        log_file = os.path.join(results_folder, "training.log")
        self.logger = create_logger(log_file=log_file, rank=not self.accelerator.is_main_process)
        self.accelerator.native_amp = amp
        """
            Initialize the model and parameters.
        """
        self.model = model
        self.train_val_forward_fn = train_val_forward_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.gradient_accumulate_every = gradient_accumulate_every
        # calculate training steps
        self.train_num_epoch = train_num_epoch
        # optimizer
        self.opt = optimizer

        # if self.accelerator.is_main_process:
        #     # save results in wandb folder if results_folder is not specified
        #     self.results_folder = Path(results_folder if results_folder
        #                                else os.path.join(self.accelerator.get_tracker('wandb', unwrap=True).dir, "../"))
        #     self.results_folder.mkdir(exist_ok=True)

        # 确保结果目录存在
        self.results_folder = Path(results_folder)
        self.checkpoint_dir = self.results_folder / "checkpoints"  # 常规检查点（按epoch）
        self.special_checkpoint_dir = self.results_folder / "special_checkpoints"  # 特殊检查点（best/last）

        if self.accelerator.is_main_process:
            self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
            self.special_checkpoint_dir.mkdir(exist_ok=True, parents=True)

        self.logger.info(f"Results will be saved to: {self.results_folder}")
        """
            Initialize the data loader.
        """
        self.cur_epoch = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt, self.scheduler, self.train_loader, self.test_loader \
            = self.accelerator.prepare(self.model, self.opt, scheduler, self.train_loader, self.test_loader)

        # 初始化最佳AUC
        global _best_auc
        if '_best_auc' not in globals():
            _best_auc = 0.0

    def save(self, name: str, max_to_keep=10):
        if not self.accelerator.is_local_main_process:
            return

        # 区分常规检查点和特殊检查点
        if name.isdigit():  # 常规epoch检查点（如name="0"）
            save_dir = self.checkpoint_dir
            save_name = save_dir / f"model-{name}.pt"
        else:  # 特殊检查点（如name="best"）
            save_dir = self.special_checkpoint_dir
            save_name = save_dir / f"model-{name}.pt"

        # 清理旧的常规检查点（仅清理checkpoint_dir下的文件）
        if name.isdigit():
            ckpt_files = glob.glob(str(save_dir / "model-[0-9]*.pt"))
            ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split('-')[-1].split('.')[0]))
            ckpt_files_to_delete = ckpt_files[:-max_to_keep]
            for ckpt_file in ckpt_files_to_delete:
                os.remove(ckpt_file)

        # 保存当前检查点（避免覆盖时重命名到special_checkpoint_dir）
        if os.path.exists(save_name):
            last_save_name = self.special_checkpoint_dir / f"model-{name}-last.pt"
            if os.path.exists(last_save_name):
                os.remove(last_save_name)
            os.rename(save_name, last_save_name)

        data = {
            'epoch': self.cur_epoch,
            'model': self.accelerator.get_state_dict(self.model),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }
        torch.save(data, save_name)
        self.logger.info(f"Saved checkpoint: {save_name}")
        if self.accelerator.is_main_process:
            self.logger.info(f"Saved checkpoint at epoch {name} to {save_name}")
            # 添加模型大小信息
            model_size = sum(p.numel() for p in self.model.parameters()) / 1e6
            self.logger.info(f"Model size: {model_size:.2f}M parameters")
#
        torch.save(data, save_name)

    def load(self, resume_path: str = None, pretrained_path: str = None):
        accelerator = self.accelerator
        device = accelerator.device

        if resume_path is not None:
            data = torch.load(resume_path, map_location=device)

            self.cur_epoch = data['epoch']
            # self.opt.load_state_dict(data['opt'])
            if exists(self.accelerator.scaler) and exists(data['scaler']):
                self.accelerator.scaler.load_state_dict(data['scaler'])

        elif pretrained_path is not None:
            data = torch.load(pretrained_path, map_location=device)
        else:
            raise ValueError('Must specify either milestone or path')
        if self.scheduler is not None:
            # step scheduler to the last epoch
            for _ in range(self.cur_epoch):
                self.scheduler.step()
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'], strict=False)

    @torch.inference_mode()
    @run_on_seed
    def val(self, model, test_data_loader, accelerator, thresholding=False, save_to=None):
        """
        validation function
        """
        global _best_mae
        if '_best_mae' not in globals():
            _best_mae = 1e10

        model.eval()
        model = accelerator.unwrap_model(model)
        device = model.device
        maes = []
        for data in tqdm(test_data_loader, disable=not accelerator.is_main_process):
            image, gt, name, image_for_post = data['image'], data['gt'], data['name'], data['image_for_post']
            gt = [np.array(x, np.float32) for x in gt]
            gt = [x / x.max() + 1e-8 for x in gt]
            image = image.to(device).squeeze(1)
            out = self.train_val_forward_fn(model, image=image, verbose=False)
            res = out["pred"].detach().cpu()
            maes += [cal_mae(g, r, thresholding, save_to, n) for g, r, n in zip(gt, res, name)]
        # gather all the results from different processes
        accelerator.wait_for_everyone()
        mae = accelerator.gather(torch.tensor(maes).mean().to(device))
        mae = mae.mean().item()
        # mae = mae_sum / test_data_loader.dataset.size
        _best_mae = min(_best_mae, mae)
        return mae, _best_mae



    @torch.inference_mode()
    @run_on_seed
    def val_time_ensemble(self, model, test_data_loader, accelerator, thresholding=False, save_to=None):
        """
        validation function
        """
        # 打印数据集长度和批次大小
        # print(f"验证数据集长度: {len(test_data_loader.dataset)}")
        # print(f"批次大小: {test_data_loader.batch_size}")

        global _best_mae
        if '_best_mae' not in globals():
            _best_mae = 1e10

        def cal_mae(gt, res, thresholding, save_to=None, n=None):
            res = res.cpu().numpy().squeeze()
            if save_to is not None:
                # # ###########################使用 os.path.splitext 分离文件名和扩展名
                filename, ext = os.path.splitext(n)
                # # ###########################将扩展名改为 .png
                new_filename = filename + '.png'
                # # # # print(new_filename)
                plt.imsave(os.path.join(save_to, new_filename), res, cmap='gray', format='png')
                # # # # # #
                # # # print(os.path.join(save_to, n),res)
            return np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        model.eval()
        model = accelerator.unwrap_model(model)
        device = model.device
        maes = defaultdict(list)
        # ensemble_maes = []
        ensemble_aucs = []
        for data in tqdm(test_data_loader, disable=not accelerator.is_main_process):
            image, gt, name, image_for_post = data['image'], data['gt'], data['name'], data['image_for_post']
            gt = [np.array(x, np.float32) for x in gt]
            gt = [x / x.max() + 1e-8 for x in gt]
            image = image.to(device).squeeze(1)
            ensem_out = self.train_val_forward_fn(model, image=image, time_ensemble=True,
                                                  gt_sizes=[g.shape for g in gt], verbose=False)
            ensem_res = ensem_out["pred"]

            # ensemble_maes += [cal_mae(g, r, thresholding, save_to, n) for g, r, n in zip(gt, ensem_res, name)]
            for g, r, n in zip(gt, ensem_res, name):
                # 保存预测图
                if save_to is not None:
                    filename, _ = os.path.splitext(n)
                    new_filename = filename + '.png'
                    save_path = os.path.join(save_to, new_filename)
                    plt.imsave(save_path, r.squeeze(), cmap='gray', format='png')

                g_uint8 = (g * 255).astype(np.uint8)
                r_uint8 = (r.numpy().squeeze() * 255).astype(np.uint8)
                auc_score = cal_auc(g_uint8, r_uint8)
                ensemble_aucs.append(auc_score)

        # gather all the results from different processes
        accelerator.wait_for_everyone()
        # ensemble_maes = accelerator.gather(torch.tensor(ensemble_maes).mean().to(device)).mean().item()
        #
        # _best_mae = min(_best_mae, ensemble_maes)
        # return ensemble_maes, _best_mae
        auc = accelerator.gather(torch.tensor(ensemble_aucs).mean().to(device)).mean().item()
        global _best_auc
        _best_auc = max(_best_auc, auc)

        return auc, _best_auc


    @torch.inference_mode()
    @run_on_seed
    def val_batch_ensemble(self, model, test_data_loader, accelerator, thresholding=False, save_to=None):
        """
        validation function
        """
        global _best_mae
        if '_best_mae' not in globals():
            _best_mae = 1e10

        model.eval()
        model = accelerator.unwrap_model(model)
        device = model.device
        ensemble_maes = []
        for data in tqdm(test_data_loader, disable=not accelerator.is_main_process):
            image, gt, name, image_for_post = data['image'], data['gt'], data['name'], data['image_for_post']
            gt = [np.array(x, np.float32) for x in gt]
            gt = [x / x.max() + 1e-8 for x in gt]
            image = image.to(device).squeeze(1)
            batch_res = []
            for i in range(5):
                ensem_out = self.train_val_forward_fn(model, image=image, time_ensemble=True, verbose=False)
                ensem_res = ensem_out["pred"].detach().cpu()
                batch_res.append(ensem_res)
            batch_res = torch.mean(torch.concat(batch_res, dim=1), dim=1, keepdim=True)
            for g, r, n in zip(gt, batch_res, name):
                ensemble_maes.append(cal_mae(g, r, thresholding, save_to, n))

        # gather all the results from different processes
        accelerator.wait_for_everyone()
        ensemble_maes = accelerator.gather(torch.tensor(ensemble_maes).mean().to(device)).mean().item()

        _best_mae = min(_best_mae, ensemble_maes)
        return ensemble_maes, _best_mae


    def train(self):
        accelerator = self.accelerator
        for epoch in range(self.cur_epoch, self.train_num_epoch):
            self.cur_epoch = epoch
            # Train
            self.model.train()
            loss_sm = SmoothedValue(window_size=10)
            with tqdm(total=len(self.train_loader), disable=not accelerator.is_main_process) as pbar:
                for data in self.train_loader:
                    with accelerator.autocast(), accelerator.accumulate(self.model):
                        # loss = fill_args_from_dict(self.train_val_forward_fn, data)(model=self.model)

                        output = fill_args_from_dict(self.train_val_forward_fn, data)(mode='train', model=self.model)
                        # print('在Trainer.train中')
                        # print('output', output)

                        # # 你之前返回的 now 是一个 dict: {'pred', 'pred_edge', 'gt', 'image'}
                        # pred = output['pred']
                        # pred_edge = output['pred_edge']
                        # gt = output['gt']
                        #
                        # print("pred", pred.shape)
                        # print("gt", gt.shape)
                        # print("pred_edge", pred_edge.shape)
                        # 调用结构损失 + 边缘损失
                        loss = output
                        # print(loss)

                        accelerator.backward(loss)
                        accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.opt.step()
                        self.opt.zero_grad()
                    loss_sm.update(loss.item())
                    pbar.set_description(
                        f'Epoch:{epoch}/{self.train_num_epoch} loss: {loss_sm.avg:.4f}({loss_sm.global_avg:.4f})')
                    self.accelerator.log({'loss': loss_sm.avg, 'lr': self.opt.param_groups[0]['lr']})
                    pbar.update()

                    # if loss_sm.count >= 20:
                    #     break
            if self.scheduler is not None:
                self.scheduler.step()

            accelerator.wait_for_everyone()
            loss_sm_gather = accelerator.gather(torch.tensor([loss_sm.global_avg]).to(accelerator.device))
            loss_sm_avg = loss_sm_gather.mean().item()
            self.logger.info(f'Epoch:{epoch}/{self.train_num_epoch} loss: {loss_sm_avg:.4f}')

            # Val
            self.model.eval()
            if (epoch + 1) % 1 == 0 or (epoch >= self.train_num_epoch * 0.7):
                # mae, best_mae = self.val_time_ensemble(self.model, self.test_loader, accelerator)
                # self.logger.info(f'Epoch:{epoch}/{self.train_num_epoch} mae: {mae:.4f}({best_mae:.4f})')
                # # #################################################################
                # accelerator.log({'mae': mae, 'best_mae': best_mae})
                # if mae == best_mae:
                #     self.save("best")

                # 原val，每轮val一次
                # auc, best_auc = self.val_time_ensemble(self.model, self.test_loader, accelerator)
                # self.logger.info(f'Epoch:{epoch}/{self.train_num_epoch} auc: {auc:.4f} (best: {best_auc:.4f})')
                # accelerator.log({'auc': auc, 'best_auc': best_auc})
                # if auc == best_auc:
                #     self.save("best")

                # 新val，改为每5个val一次
                # if (epoch + 1) % 1 == 0:
                # epoch>100 val
                if epoch >= 0 :
                    auc, best_auc = self.val_time_ensemble(self.model, self.test_loader, accelerator)
                    self.logger.info(f'Epoch:{epoch}/{self.train_num_epoch} auc: {auc:.4f} (best: {best_auc:.4f})')
                    accelerator.log({'auc': auc, 'best_auc': best_auc})
                    if auc == best_auc:
                        self.save("best")
                # ###############

            # 保存常规检查点
            if epoch % 5 == 0:  # 每5个epoch保存一次
                self.save(str(epoch))  # 将整数转换为字符串

            self.save(str(self.cur_epoch))  # cur_epoch 是整数，转换为字符串

            # # Visualize
            # with torch.inference_mode():
            #     if accelerator.is_main_process:
            #         model = self.accelerator.unwrap_model(self.model)
            #         for tracker in accelerator.trackers:
            #             if tracker.name == "wandb":
            #                 out = fill_args_from_dict(self.train_val_forward_fn, data)(model=model,
            #                                                                            verbose=False)
            #                 tracker.log(
            #                     {'pred-img-mask':
            #                          [wandb.Image(o[0, :, :]) for o in out.values()]
            #                      })

            # 可视化结果保存到本地
            if self.accelerator.is_main_process and epoch % 10 == 0:
                viz_dir = self.results_folder / "visualizations" / f"epoch_{epoch}"
                viz_dir.mkdir(parents=True, exist_ok=True)

                model = self.accelerator.unwrap_model(self.model)
                with torch.inference_mode():
                    # 选择少量样本进行可视化
                    sample_data = next(iter(self.test_loader))
                    sample_data['image'] = sample_data['image'].squeeze(1)  # 压缩第1维
                    out = fill_args_from_dict(self.train_val_forward_fn, sample_data)(model=model, verbose=False)

                    # 修正后代码（添加 squeeze 去除通道维度）
                    for i, pred in enumerate(out["pred"][:5]):
                        # 假设 pred 形状为 (1, H, W)，去除第0维的通道维度
                        pred_2d = pred.squeeze(0).cpu().numpy()  # 或 pred.cpu().numpy().squeeze(0)
                        # print(f"修正后 pred 形状: {pred_2d.shape}")  # 应输出 (H, W)
                        plt.imsave(viz_dir / f"pred_{i}.png", pred_2d, cmap='gray')

                        # # 3FenZhi新增：保存边缘预测结果（关键修改）
                        # for i, pred_edge in enumerate(out["pred_edge"][:5]):  # 取前5个样本（与pred对应）
                        #     pred_edge_2d = pred_edge.squeeze(0).cpu().numpy()  # 同样去除通道维度
                        #     plt.imsave(viz_dir / f"pred_edge_{i}.png", pred_edge_2d, cmap='gray')  # 文件名区分pred_edge

            accelerator.wait_for_everyone()
        self.logger.info('training complete')
        accelerator.end_training()
