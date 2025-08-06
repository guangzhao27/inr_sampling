from models import Siren, PEMLP
from util.logger import log
from util.tensorboard import writer
from util.plotter import plotter
from util.misc import fix_seed, calc_intersec
from util import io

from trainer.img_trainer import ImageTrainer

from components.ssim import compute_ssim_loss as compute_ssim
from components.laplacian import compute_laplacian_loss as compute_laplacian
from components.lmc import LMC
from components.nmt import NMT
from components.nmt import mt_scheduler_factory
from components.expansive import ExpansiveSupervision as ES
from components.egra import EGRA

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import trange


class Sampler(object):
    def __init__(self, args):
        self.args = args
        self._st = self.args.strategy
        self.use_ratio_scheduler = mt_scheduler_factory(self.args.sample_num_schedular)
        self.book = {}
 
    def _reset_rng(self):
        generator = torch.Generator()
        seed = generator.seed()

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _recover_rng(self):
        fix_seed(self.args.seed)
    
    def _init_sampler(self):
    
        elif self.args.strategy == "evos":
            self._evos_init()

    def _evos_init(self):
        if self.args.lap_coff > 0 or self.args.crossover_method != "no":
            self.cached_gt_lap = compute_laplacian(self.input_img).squeeze()

    def _get_cur_use_ratio(self, epoch):
        return self.use_ratio_scheduler(
            epoch, self.args.num_epochs, self.args.use_ratio
        )

    def _sampler_get_coords_gt(self, epoch):
        coords, gt = self.full_coords, self.full_gt
        self.cur_use_ratio = self._get_cur_use_ratio(epoch)

        self._reset_rng()

        _st = self.args.strategy

        elif _st == "evos":
            if self._evos_is_fitness_eval_iter(epoch):
                return coords, gt
            else:
                selection_mask = self._evos_get_selection_mask(epoch)
                _coords = self.full_coords[selection_mask]
                _gt = self.full_gt[selection_mask]
                return _coords, _gt

        else:
            raise NotImplementedError

        self._recover_rng()
        return _coords, _gt

    def _sampler_compute_loss(self, pred, gt, epoch):
        _st = self.args.strategy
        mse = self.compute_mse(pred, gt)
        if _st == "evos":
            if self._evos_is_fitness_eval_iter(epoch):

                self._evos_frequency_aware_crossover(pred, gt, epoch) # crossover
                return self._evos_cross_frequency_loss(mse, pred)
            else:
                if self.args.lap_coff <= 0 or epoch > self.args.use_laplace_epoch:
                    return mse
                else:
                    profile_pred = self.book["freeze_profile_pred"]
                    _mask = self.book["freeze_mask"]
                    pseudo_full_pred = profile_pred.clone()

                    indices = torch.arange(_mask.shape[0], device=pred.device)[~_mask]
                    pseudo_full_pred[indices] = pred

                    r_img = self.reconstruct_img(pseudo_full_pred)
                    lap_loss = (
                        F.mse_loss(
                            compute_laplacian(r_img).squeeze(),
                            self.cached_gt_lap,
                            reduction="none",
                        )
                        .flatten()[~_mask]
                        .mean()
                    )
                    
                    return mse + self.args.lap_coff * lap_loss
        else:
            return mse

    def _evos_get_mutation_ratio(self, epoch):
        if self.args.mutation_method == "constant":
            return self.args.init_mutation_ratio * self.args.use_ratio
        elif self.args.mutation_method == "linear":
            _start = self.args.init_mutation_ratio
            _end = self.args.end_mutation_ratio  # max = 1
            ratio = _start + ((_end - _start) / self.args.num_epochs) * epoch
            return ratio * self.args.use_ratio
        elif self.args.mutation_method == "exp":
            _start = self.args.init_mutation_ratio
            _end = self.args.end_mutation_ratio
            _lamda = -np.log(_end / _start) / self.args.num_epochs
            ratio = _start * np.exp(-_lamda * epoch)
            return ratio * self.args.use_ratio
        else:
            raise NotImplementedError

    def _evos_get_selection_mask(self, epoch):
        mutation_ratio = self._evos_get_mutation_ratio(epoch)
        first_select_ratio = self.cur_use_ratio - mutation_ratio
        first_select_num = int(first_select_ratio * self.sample_num)

        sorted_map_index = self.book["sorted_map_index"]
        first_select_indices = sorted_map_index[-first_select_num:]

        # Augmented Unbiased Mutation
        mutation_num = int(mutation_ratio * self.sample_num)
        remain_indices = sorted_map_index[:-first_select_num]
        sample_index = torch.randperm(remain_indices.shape[0], device=self.device)[
            :mutation_num
        ]
        mutation_indicies = remain_indices[sample_index]

        selected_indices = torch.cat([first_select_indices, mutation_indicies])

        _mask = torch.ones(self.sample_num, dtype=torch.bool, device=self.device)
        _mask[selected_indices] = False
        self.book["freeze_mask"] = _mask

        selection_mask = torch.zeros(
            self.sample_num, dtype=torch.bool, device=self.device
        )
        selection_mask[selected_indices] = True
        return selection_mask

    def _evos_is_fitness_eval_iter(self, epoch):
        _cur_interval = self._evos_get_cur_interval(epoch)
        return epoch % _cur_interval == 1

    def _evos_get_cur_interval(self, epoch):
        if self.args.profile_interval_method == "fixed":
            return self.args.init_interval
        elif self.args.profile_interval_method == "lin_dec":
            _start = self.args.init_interval
            _end = self.args.end_interval
            _cur_interval = _start + ((_end - _start) / self.args.num_epochs) * epoch
            return int(_cur_interval)

    def _evos_frequency_aware_crossover(self, pred, gt, epoch): 
        error_map = F.mse_loss(pred, gt, reduction="none").mean(1)
        if self.args.crossover_method == "add":
            r_img = self.reconstruct_img(pred)
            laplace_map = F.mse_loss(
                compute_laplacian(r_img).squeeze(), self.cached_gt_lap, reduction="none"
            )
            cross_lap_coff = self.args.lap_coff if self.args.lap_coff > 0 else 1e-5
            error_map = error_map + cross_lap_coff * laplace_map.flatten()
        elif self.args.crossover_method == "no":
            pass

        if self.args.profile_guide == "value":
            sorted_map_index = torch.argsort(error_map.flatten())
        elif self.args.profile_guide == "diff_1":
            # to deprecated ...
            last_error_map = self.book.get("error_map", None)
            if last_error_map is None:
                last_error_map = torch.zeros_like(error_map)
            guidance_map = torch.abs(error_map - last_error_map)
            sorted_map_index = torch.argsort(guidance_map.flatten())
        else:
            raise NotImplementedError

        self.book["freeze_profile_pred"] = pred.detach()
        self.book["error_map"] = error_map.detach()
        self.book["sorted_map_index"] = sorted_map_index

        if self.args.crossover_method == "select":
            r_img = self.reconstruct_img(pred)
            laplace_map = F.mse_loss(
                compute_laplacian(r_img).squeeze(), self.cached_gt_lap, reduction="none"
            )
            cross_lap_coff = self.args.lap_coff if self.args.lap_coff > 0 else 1e-5
            laplace_error_map = cross_lap_coff * laplace_map.flatten()
            sorted_lap_map_index = torch.argsort(laplace_error_map.flatten())
            self.book["sorted_lap_map_index"] = sorted_lap_map_index

            mutation_ratio = self._evos_get_mutation_ratio(epoch)
            freeze_ratio = 1 - self.cur_use_ratio + mutation_ratio

            freezed_num = int(freeze_ratio * self.sample_num)
            selected_num = self.sample_num - freezed_num

            l2_error_selected_index = sorted_map_index[-selected_num:]
            lap_error_selected_index = sorted_lap_map_index[-selected_num:]
            isin = torch.isin(l2_error_selected_index, lap_error_selected_index)

            selected_index = l2_error_selected_index[isin]

            remain_num = selected_num - selected_index.shape[0]
            l2_remain_index = l2_error_selected_index[~isin]
            isin2 = torch.isin(lap_error_selected_index, l2_error_selected_index)
            lap_remain_index = lap_error_selected_index[~isin2]

            l2_remain_num = int(
                remain_num
                * (error_map.mean() / (laplace_error_map.mean() + error_map.mean()))
            )
            l2_remain_num = min(l2_remain_num, l2_remain_index.shape[0])
            lap_remain_num = remain_num - l2_remain_num
            all_selected_index = torch.cat(
                [
                    lap_remain_index[-lap_remain_num:],
                    l2_remain_index[-l2_remain_num:],
                    selected_index,
                ]
            )
            all_remain_index = sorted_map_index[
                ~torch.isin(sorted_map_index, all_selected_index)
            ]
            select_sorted_index = torch.cat([all_remain_index, all_selected_index])
            self.book["sorted_map_index"] = select_sorted_index

    def _evos_cross_frequency_loss(self, cur_loss, pred):
        if self.args.lap_coff > 0:
            r_img = self.reconstruct_img(pred)
            lap_loss = F.mse_loss(
                compute_laplacian(r_img).squeeze(), self.cached_gt_lap
            )
            cur_loss += self.args.lap_coff * lap_loss
        return cur_loss