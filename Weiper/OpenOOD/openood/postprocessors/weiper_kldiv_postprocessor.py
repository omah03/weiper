import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from typing import Any, List

from .base_postprocessor import BasePostprocessor
from .weiper_kldiv.utils import (
    calculate_WeiPerKLDiv_score,
    calculate_weiper_space,
)
#from aggregator_mdcdf import MDCDFAggregator  # aggregator if aggregator_mode="mdcdf"


class WeiPerKLDivPostprocessor(BasePostprocessor):
    """
    This class supports:
      - aggregator_mode in {"none","sum","weighted_sum","mdcdf"}
      - If aggregator_mode="none" + layer_names=[4], single-layer original WeiPer logic.
      - If aggregator_mode="none" + multiple layers => default mean across layers.
      - If aggregator_mode="sum"/"weighted_sum" => no aggregator needed; sum or weighted-sum across layers.
      - If aggregator_mode="mdcdf" => multi-dim aggregator with NPZ caching logic.
    """

    def __init__(self, config):
        super().__init__(config)

        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep

        # ~~~~~~~~~ WeiPer / KLDIV hyperparams ~~~~~~~~~
        self.lambda_1 = self.args.lambda_1
        self.lambda_2 = self.args.lambda_2
        self.exact_minmax = self.args.exact_minmax
        self.smoothing = self.args.smoothing
        self.smoothing_perturbed = self.args.smoothing_perturbed
        self.n_bins = self.args.n_bins
        self.perturbation_distance = self.args.perturbation_distance
        self.n_repeats = self.args.n_repeats
        self.n_samples_for_setup = self.args.n_samples_for_setup

        # aggregator_mode => "none","sum","weighted_sum","mdcdf"
        self.aggregator_mode = getattr(self.args, "aggregator_mode", "none")

        # layer_names => which layers to use
        self.layer_names = getattr(self.args, "layer_names", [3,4])
        if not isinstance(self.layer_names, list):
            self.layer_names = [self.layer_names]

        # Weighted-sum approach
        self.layer_weights = getattr(self.args, "layer_weights", None)
        if self.layer_weights is not None:
            total = sum(self.layer_weights)
            if total <= 0:
                raise ValueError("[ERROR] sum of layer_weights <=0!")
            # normalize
            self.layer_weights = [w / total for w in self.layer_weights]
            print("[DEBUG] aggregator_mode=%s => normalized layer_weights=%s"
                  % (self.aggregator_mode, self.layer_weights))
        else:
            print("[DEBUG] aggregator_mode=%s => no layer_weights (default mean if multi-layers)."
                  % self.aggregator_mode)

        # MDCDF aggregator
        self.method = getattr(self.args, "mdcdf_method", "kde")
        self.invert_cdf = getattr(self.args, "invert_cdf", False)
        self.agg_debug = getattr(self.args, "agg_debug", False)
        self.agg_kwargs = getattr(self.args, "agg_kwargs", {})
        self.mdcdf_aggregator = None

        # For storing each layer’s distribution
        self.layer_train_info = {}
        # For single-layer approach: we store penultimate logic in self.train_densities
        self.train_densities = None
        self.train_min_max = None
        self.W_tilde = None

        # NPZ logic
        self.master_layers = [1, 2, 3, 4]
        self.master_npz_filename = "id_scores_layer_1_2_3_4.npz"
        self.id_score_matrix = None

        print("[DEBUG] WeiPerKLDivPostprocessor constructed. aggregator_mode=%s" % self.aggregator_mode)
        print("        layer_names:", self.layer_names)
        if self.layer_weights:
            print("        layer_weights:", self.layer_weights)
        print()

    def flatten_features(self, feat: torch.Tensor) -> torch.Tensor:
        """ Flatten NxCxHxW => NxD """
        return feat.view(feat.size(0), -1)

    @torch.no_grad()
    def setup(
        self,
        net: nn.Module,
        id_loader_dict,
        ood_loader_dict,
        id_name="cifar10",
        valid_num=None,
        aps=None,
        use_cache=False,
        hyperparameter_search=False,
        latents_loader=None,
        **kwargs,
    ):
        """
        1) If aggregator_mode='none' AND layer_names in {4} => single-layer original WeiPer penultimate logic
        2) Otherwise => multi-layer logic. Possibly sum/weighted_sum or aggregator='mdcdf'
        """
        print("[DEBUG] Entering WeiPerKLDivPostprocessor.setup")
        net.eval()
        device = next(iter(net.parameters())).device
        train_dl = id_loader_dict["train"] if latents_loader is None else latents_loader

        # Case 1) aggregator_mode='none' + single-layer penultimate => do old approach
        if self.aggregator_mode == "none" and len(self.layer_names) == 1 and self.layer_names[0] == 4:
            print("[DEBUG] aggregator_mode='none' & single-layer=[4] => single-layer penultimate approach.")
            self._setup_singlelayer_original(net, train_dl, device)
            return

        # Otherwise => multi-layer aggregator approach
        # always ensure layer 4 included if not present
        if 4 not in self.layer_names:
            self.layer_names.append(4)

        # If exact_minmax => gather pen-latent min/max
        if self.exact_minmax:
            print("[DEBUG] exact_minmax=True => scanning penultimate-latents.")
            self._scan_penultimate_minmax(net, train_dl, device)
        else:
            print("[DEBUG] exact_minmax=False => skipping pen-latent scanning.")
            self.train_min_max = None
            self.W_tilde = None

        # Build each layer’s distribution
        self._build_multi_layer_distributions(net, train_dl, device)

        # If aggregator_mode in {'none','sum','weighted_sum'}, we skip multi-d aggregator
        if self.aggregator_mode in ["none", "sum", "weighted_sum"]:
            print("[DEBUG] aggregator_mode=%s => no multi-dim aggregator to fit." % self.aggregator_mode)
            return

        # aggregator_mode=="mdcdf" => build or load aggregator ID-scores
        print("[DEBUG] aggregator_mode='mdcdf' => building/loading aggregator NPZ.")
        # self.mdcdf_aggregator = MDCDFAggregator(
        #     method=self.method,
        #     invert_cdf=self.invert_cdf,
        #     debug_enabled=self.agg_debug,
        #     **self.agg_kwargs
        # )

        # define the file name for the user-layers
        sorted_lay = sorted(self.layer_names)
        layer_str = "_".join(map(str, sorted_lay))
        fname = f"id_scores_layer_{layer_str}.npz"

        # check if full master => if yes, can load or slice
        if sorted_lay == self.master_layers:
            if os.path.exists(self.master_npz_filename):
                print("[DEBUG] Found master =>", self.master_npz_filename)
                loaded = np.load(self.master_npz_filename)
                self.id_score_matrix = loaded["scores"]  # shape [N,4]
                print("[DEBUG] aggregator ID-scores shape =>", self.id_score_matrix.shape)
            else:
                # build aggregator data for [1,2,3,4]
                print("[DEBUG] No master => building aggregator data for [1,2,3,4]")
                self.id_score_matrix = self._build_id_scores_matrix(net, train_dl, device, sorted_lay)
                np.savez(self.master_npz_filename, scores=self.id_score_matrix)
        else:
            # partial => see if we can splice from master
            if os.path.exists(self.master_npz_filename):
                print("[DEBUG] Splicing columns from master =>", self.master_npz_filename)
                data = np.load(self.master_npz_filename)
                full_scores = data["scores"] # shape [N,4]
                col_map = {1:0,2:1,3:2,4:3}
                chosen_cols = [col_map[l] for l in sorted_lay]
                sub = full_scores[:, chosen_cols]
                self.id_score_matrix = sub
                print("[DEBUG] sub-scores shape =>", sub.shape)
            else:
                # check a smaller subset file
                if os.path.exists(fname):
                    print("[DEBUG] Found aggregator subset =>", fname)
                    data = np.load(fname)
                    self.id_score_matrix = data["scores"]
                    print("[DEBUG] aggregator ID-scores shape =>", self.id_score_matrix.shape)
                else:
                    print("[DEBUG] No aggregator subset => building from scratch =>", sorted_lay)
                    self.id_score_matrix = self._build_id_scores_matrix(net, train_dl, device, sorted_lay)
                    np.savez(fname, scores=self.id_score_matrix)

        if self.id_score_matrix is None:
            raise RuntimeError("[ERROR] aggregator_mode='mdcdf' but no aggregator ID-scores found.")

        # aggregator fit
        print("[DEBUG] aggregator => fit shape =>", self.id_score_matrix.shape)
        self.mdcdf_aggregator.fit(self.id_score_matrix)
        print("[DEBUG] aggregator fit done => method=", self.method)

    def _setup_singlelayer_original(self, net, train_dl, device):
        """
        The old style single-layer penultimate approach => store in self.train_densities
        """
        print("[DEBUG] _setup_singlelayer_original => single-layer penultimate logic.")
        latmin, latmax = float("inf"), float("-inf")
        weimin, weimax = float("inf"), float("-inf")

        # if exact_minmax => do scanning
        if self.exact_minmax:
            for i,batch in enumerate(train_dl):
                data = batch["data"].to(device)
                _, feat = net(data, return_feature=True)
                weip, self.W_tilde = calculate_weiper_space(
                    net, feat, self.W_tilde, device,
                    self.perturbation_distance, self.n_repeats
                )
                latmin = min(latmin, feat.min().item())
                latmax = max(latmax, feat.max().item())
                weimin = min(weimin, weip.min().item())
                weimax = max(weimax, weip.max().item())
            self.train_min_max = ((latmin, latmax),(weimin, weimax))
            print("[DEBUG] single-layer => pen-latent scanning =>", self.train_min_max)
        else:
            print("[DEBUG] single-layer => exact_minmax=False => skipping scanning.")
            self.train_min_max=None
            self.W_tilde=None

        # build distributions
        accumA=None
        accumB=None
        count=0
        for i,batch in enumerate(train_dl):
            data = batch["data"].to(device)
            _, feat = net(data, return_feature=True)
            dA, dB, upd_mm, upd_wt = calculate_WeiPerKLDiv_score(
                net,
                feat,
                lambda_1=self.lambda_1,
                lambda_2=self.lambda_2,
                n_bins=self.n_bins,
                n_repeats=self.n_repeats,
                smoothing=self.smoothing,
                smoothing_perturbed=self.smoothing_perturbed,
                perturbation_distance=self.perturbation_distance,
                train_densities=None,
                train_min_max=self.train_min_max,
                device=device,
                W_tilde=self.W_tilde
            )
            if accumA is None:
                accumA = dA
                accumB = dB
            else:
                accumA += dA
                accumB += dB
            count+=1
            self.train_min_max=upd_mm
            self.W_tilde=upd_wt

        # average
        accumA = (accumA / count).mean(dim=-1, keepdim=True)
        accumB = (accumB / count).mean(dim=-1, keepdim=True)
        self.train_densities = (accumA, accumB)
        print("[DEBUG] single-layer penultimate => final shape =>", accumA.shape, accumB.shape)

    def _scan_penultimate_minmax(self, net, train_dl, device):
        """
        Utility to fill self.train_min_max by scanning pen-lat & weiper-lat
        """
        penMin, penMax = float("inf"), float("-inf")
        weiMin, weiMax = float("inf"), float("-inf")
        self.W_tilde=None
        for i, batch in enumerate(train_dl):
            data=batch["data"].to(device)
            _, feat_list = net(data, return_feature_list=True)
            pen = self.flatten_features(feat_list[4])
            if i==0:
                self.W_tilde=None
            weip, self.W_tilde = calculate_weiper_space(
                net, pen, self.W_tilde, device,
                self.perturbation_distance, self.n_repeats
            )
            penMin = min(penMin, pen.min().item())
            penMax = max(penMax, pen.max().item())
            weiMin = min(weiMin, weip.min().item())
            weiMax = max(weiMax, weip.max().item())
        self.train_min_max = ((penMin, penMax),(weiMin, weiMax))
        print("[DEBUG] _scan_penultimate_minmax => final =>", self.train_min_max)

    def _build_multi_layer_distributions(self, net, train_dl, device):
        """
        Build distributions for each layer in self.layer_names => store in self.layer_train_info
        """
        accum_dict = {}
        for l_idx in self.layer_names:
            key='penultimate' if l_idx==4 else l_idx
            accum_dict[key] = {"dens":None, "weiper_dens":None, "count":0}

        for i,batch in enumerate(train_dl):
            data = batch["data"].to(device)
            _, feat_list = net(data, return_feature_list=True)
            for l_idx in self.layer_names:
                key='penultimate' if l_idx==4 else l_idx
                lat = self.flatten_features(feat_list[l_idx])
                is_pen = (l_idx==4)
                densA, densB, upMM, upWT = calculate_WeiPerKLDiv_score(
                    net,
                    lat,
                    lambda_1=self.lambda_1,
                    lambda_2=self.lambda_2,
                    n_bins=self.n_bins,
                    n_repeats=(self.n_repeats if is_pen else 1),
                    smoothing=self.smoothing,
                    smoothing_perturbed=self.smoothing_perturbed,
                    perturbation_distance=(self.perturbation_distance if is_pen else 0),
                    train_densities=None,
                    train_min_max=(self.train_min_max if is_pen else None),
                    W_tilde=(self.W_tilde if is_pen else None),
                    device=device,
                )
                if accum_dict[key]["dens"] is None:
                    accum_dict[key]["dens"] = densA
                    accum_dict[key]["weiper_dens"] = densB
                else:
                    accum_dict[key]["dens"] += densA
                    accum_dict[key]["weiper_dens"] += densB
                accum_dict[key]["count"] +=1
                if is_pen:
                    self.train_min_max=upMM
                    self.W_tilde=upWT

        for key, vals in accum_dict.items():
            c=vals["count"]
            dA = (vals["dens"]/c).mean(dim=-1, keepdim=True)
            dB = (vals["weiper_dens"]/c).mean(dim=-1, keepdim=True)
            self.layer_train_info[key] = {
                "train_densities": (dA, dB),
                "train_min_max": (self.train_min_max if key=="penultimate" else None),
                "W_tilde": (self.W_tilde if key=="penultimate" else None)
            }
        print("[DEBUG] multi-layer distribution =>", list(self.layer_train_info.keys()))

    def _build_id_scores_matrix(self, net, train_dl, device, sorted_layers:List[int]):
        """
        aggregator usage => gather final OOD for each layer. shape => [N, #layers].
        """
        all_rows=[]
        if self.n_samples_for_setup>0:
            limit = min(len(train_dl.dataset), self.n_samples_for_setup)
        else:
            limit = len(train_dl.dataset)

        scount=0
        train_iter = iter(train_dl)
        with torch.no_grad():
            while True:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    break
                data = batch["data"].to(device)
                _, flist = net(data, return_feature_list=True)
                bs = data.shape[0]
                for bidx in range(bs):
                    if scount>=limit:
                        break
                    row=[]
                    for l_idx in sorted_layers:
                        key='penultimate' if l_idx==4 else l_idx
                        lat = self.flatten_features(flist[l_idx])
                        lat_single=lat[bidx:bidx+1]
                        scval = self._compute_one_layer_ood_score(net, lat_single, l_idx, device)
                        row.append(scval.item())
                    all_rows.append(row)
                    scount+=1
                if scount>=limit:
                    break
        arr = np.array(all_rows, dtype=np.float32)
        return arr

    def _compute_one_layer_ood_score(self, net, lat_single, layer_idx, device):
        """
        partial logic => returns final OOD for that sample, layer
        """
        key='penultimate' if layer_idx==4 else layer_idx
        tdens, twdens = self.layer_train_info[key]["train_densities"]
        tmm = self.layer_train_info[key]["train_min_max"]
        tW = self.layer_train_info[key]["W_tilde"]
        is_pen = (layer_idx==4)
        conf=calculate_WeiPerKLDiv_score(
            net,
            lat_single,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            n_bins=self.n_bins,
            n_repeats=(self.n_repeats if is_pen else 1),
            smoothing=self.smoothing,
            smoothing_perturbed=self.smoothing_perturbed,
            perturbation_distance=(self.perturbation_distance if is_pen else 0.0),
            train_densities=(tdens, twdens),
            train_min_max=(tmm if is_pen else None),
            W_tilde=(tW if is_pen else None),
            device=device,
        )
        return conf.squeeze()

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, latents: torch.Tensor=None):
        """
        Depending on aggregator_mode => {none,sum,weighted_sum,mdcdf},
        we gather multi-layer OOD scores => final aggregator or sum or mean, etc.
        """
        print("[DEBUG] Entering WeiPerKLDivPostprocessor.postprocess => aggregator_mode=%s" % self.aggregator_mode)
        net.eval()
        device = next(iter(net.parameters())).device

        # if aggregator_mode='none' & single-layer => old chunk approach
        if self.aggregator_mode=="none" and len(self.layer_names)==1 and self.layer_names[0]==4:
            return self._postprocess_singlelayer_original(net, data, latents)

        # Otherwise => gather multi-layer OOD
        if latents is not None:
            print("[DEBUG] latents provided => ignoring multi-layers except penultimate.")
            pred=None
            feats=[None,None,None,None, latents]
            used_layers=["penultimate"]  # only penultimate is valid
            N= latents.shape[0]
        else:
            out, flist = net(data.to(device), return_feature_list=True)
            pred = torch.argmax(F.softmax(out, dim=1), dim=1)
            # ensure penultimate
            used_layers = list(self.layer_names)
            if 4 not in used_layers:
                used_layers.append(4)
            feats=flist
            N=data.shape[0]
            print("[DEBUG] aggregator => used_layers =>", used_layers)

        # chunk-based approach
        chunk_sz=512
        allres=[]
        for start_idx in range(0,N,chunk_sz):
            end_idx=min(start_idx+chunk_sz,N)
            length=end_idx - start_idx
            chunk_scores = np.zeros((length, len(used_layers)), dtype=np.float32)
            for li,l_idx in enumerate(used_layers):
                key='penultimate' if l_idx==4 else l_idx
                if l_idx==4:
                    lat_full=self.flatten_features(feats[4])
                else:
                    lat_full=self.flatten_features(feats[l_idx])
                lat_chunk=lat_full[start_idx:end_idx]
                is_pen=(l_idx==4)

                sc=calculate_WeiPerKLDiv_score(
                    net,
                    lat_chunk,
                    lambda_1=self.lambda_1,
                    lambda_2=self.lambda_2,
                    n_bins=self.n_bins,
                    n_repeats=(self.n_repeats if is_pen else 1),
                    smoothing=self.smoothing,
                    smoothing_perturbed=self.smoothing_perturbed,
                    perturbation_distance=(self.perturbation_distance if is_pen else 0.0),
                    train_densities=self.layer_train_info[key]["train_densities"],
                    train_min_max=self.layer_train_info[key]["train_min_max"],
                    W_tilde=self.layer_train_info[key]["W_tilde"],
                    device=device,
                )
                chunk_scores[:,li] = sc.cpu().numpy()
            allres.append(chunk_scores)
        score_mat = np.concatenate(allres, axis=0)  # shape [N, #used_layers]

        # aggregator
        if self.aggregator_mode=="sum":
            print("[DEBUG] aggregator_mode='sum' => sum across layers.")
            final_np = score_mat.sum(axis=1)
        elif self.aggregator_mode=="weighted_sum":
            print("[DEBUG] aggregator_mode='weighted_sum'.")
            if not self.layer_weights or len(self.layer_weights)!=score_mat.shape[1]:
                raise ValueError("[ERROR] aggregator_mode='weighted_sum' => mismatch # of layers vs layer_weights!")
            final_np = (score_mat * np.array(self.layer_weights)).sum(axis=1)
        elif self.aggregator_mode=="none":
            # multiple layers but aggregator='none'? => mean
            print("[DEBUG] aggregator_mode='none' => multi-layers => using mean.")
            final_np = score_mat.mean(axis=1)
        elif self.aggregator_mode=="mdcdf":
            print("[DEBUG] aggregator_mode='mdcdf' => aggregator scoring => shape", score_mat.shape)
            final_np = self.mdcdf_aggregator.score(score_mat)
        else:
            raise ValueError("[ERROR] aggregator_mode not recognized =>", self.aggregator_mode)

        final_th = torch.from_numpy(final_np).float().to(device)
        return pred, final_th

    def _postprocess_singlelayer_original(self, net, data:Any, latents:torch.Tensor=None):
        """
        The single-layer chunk approach => aggregator_mode='none' & layer_names=[4].
        """
        print("[DEBUG] single-layer penultimate postprocess => aggregator_mode='none' & single-layer")
        device=next(iter(net.parameters())).device
        if latents is not None:
            feat= latents.to(device)
            pred=None
        else:
            out, feat = net(data.to(device), return_feature=True)
            pred = torch.argmax(F.softmax(out,dim=1),dim=1)

        chunk_sz=512
        all_chunks=[]
        for start_idx in range(0, feat.shape[0], chunk_sz):
            end_idx=min(start_idx+chunk_sz, feat.shape[0])
            cfeat= feat[start_idx:end_idx]
            conf= calculate_WeiPerKLDiv_score(
                net,
                cfeat,
                lambda_1=self.lambda_1,
                lambda_2=self.lambda_2,
                n_bins=self.n_bins,
                n_repeats=self.n_repeats,
                smoothing=self.smoothing,
                smoothing_perturbed=self.smoothing_perturbed,
                perturbation_distance=self.perturbation_distance,
                train_densities=self.train_densities,
                train_min_max=self.train_min_max,
                device=device,
                W_tilde=self.W_tilde,
            )
            all_chunks.append(conf)
        final_conf = torch.cat(all_chunks, dim=0)
        return pred, final_conf

    def set_hyperparam(self, hyperparam: list):
        print("[DEBUG] Setting hyperparameters =>", hyperparam)
        self.lambda_1 = hyperparam[0]
        self.lambda_2 = hyperparam[1]
        self.smoothing = hyperparam[2]
        self.smoothing_perturbed = hyperparam[3]
        self.n_bins = hyperparam[4]
        self.perturbation_distance = hyperparam[5]

    def get_hyperparam(self):
        return (
            self.lambda_1,
            self.lambda_2,
            self.smoothing,
            self.smoothing_perturbed,
            self.n_bins,
            self.perturbation_distance,
        )
