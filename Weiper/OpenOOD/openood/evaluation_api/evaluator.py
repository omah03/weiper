from typing import Callable, List, Type

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from openood.evaluators.metrics import compute_all_metrics
from openood.postprocessors import BasePostprocessor
from openood.networks.ash_net import ASHNet
from openood.networks.react_net import ReactNet
from openood.postprocessors.weiper_kldiv.imglist_generator import generate_imglist

from .datasets import DATA_INFO, data_setup, get_id_ood_dataloader
from .postprocessor import get_postprocessor
from .preprocessor import get_default_preprocessor


class Evaluator:
    def __init__(
        self,
        net: nn.Module,
        id_name: str,
        data_root: str = "./data",
        config_root: str = "./configs",
        preprocessor: Callable = None,
        postprocessor_name: str = None,
        postprocessor: Type[BasePostprocessor] = None,
        batch_size: int = 200,
        shuffle: bool = False,
        num_workers: int = 4,
        cached_dir: str = "./cache",
        use_cache: bool = False,
        APS_mode: bool = True,
        verbose: bool = False,
        **postpc_kwargs,
    ) -> None:
        """
        A unified, easy-to-use API for evaluating (most) discriminative OOD detection methods,
        with modifications to only handle "near" OOD (no "far" OOD).
        """
        # check the arguments
        if postprocessor_name is None and postprocessor is None:
            raise ValueError("Please pass postprocessor_name or postprocessor")
        if postprocessor_name is not None and postprocessor is not None:
            if verbose:
                print("Postprocessor_name is ignored because postprocessor is passed")
        if id_name not in DATA_INFO:
            raise ValueError(f"Dataset [{id_name}] is not supported")
        self.postprocessor_name = postprocessor_name
        # get data preprocessor
        if preprocessor is None:
            preprocessor = get_default_preprocessor(id_name)

        # set up config root
        if config_root is None:
            filepath = os.path.dirname(os.path.abspath(__file__))
            config_root = os.path.join(*filepath.split("/")[:-2], "configs")

        # get postprocessor
        if postprocessor is None:
            postprocessor = get_postprocessor(config_root, postprocessor_name, id_name)
        if not isinstance(postprocessor, BasePostprocessor):
            raise TypeError("postprocessor should inherit BasePostprocessor in OpenOOD")

        # load data
        data_setup(data_root, id_name)
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
        }

        if postprocessor_name == "weiper_kldiv" and id_name == "imagenet":
            generate_imglist(postprocessor.n_samples_for_setup, data_dir=data_root)
        dataloader_dict = get_id_ood_dataloader(
            id_name,
            data_root,
            preprocessor,
            postprocessor_name=postprocessor_name,
            **loader_kwargs,
        )

        # wrap base model to work with certain postprocessors
        if postprocessor_name == "react":
            net = ReactNet(net)
        elif postprocessor_name == "ash":
            net = ASHNet(net)

        # postprocessor setup
        if postprocessor_name == "nac":
            postprocessor.setup(
                net,
                dataloader_dict["id"],
                dataloader_dict["ood"],
                use_cache=False,
                id_name=id_name,
                **postpc_kwargs,
            )
        else:
            postprocessor.setup(net, dataloader_dict["id"], dataloader_dict["ood"])

        self.id_name = id_name
        self.net = net
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.dataloader_dict = dataloader_dict

        # Instead of storing near/far, we'll only store "near" in "ood"
        self.metrics = {"id_acc": None, "csid_acc": None, "ood": None, "fsood": None}

        # We skip the "far" dictionary references:
        self.scores = {
            "id": {"train": None, "val": None, "test": None},
            "csid": {k: None for k in dataloader_dict["csid"].keys()},
            "ood": {
                "val": None,
                "near": {k: None for k in dataloader_dict["ood"]["near"].keys()},
                # "far": {k: None for k in dataloader_dict["ood"]["far"].keys()},
                # removed "far"
            },
            "id_preds": None,
            "id_labels": None,
            "csid_preds": {k: None for k in dataloader_dict["csid"].keys()},
            "csid_labels": {k: None for k in dataloader_dict["csid"].keys()},
        }

        # perform hyperparameter search if have not done so
        if (
            self.postprocessor.APS_mode
            and not self.postprocessor.hyperparam_search_done
            and APS_mode
        ):
            self.hyperparam_search(verbose)

        self.net.eval()

    def _classifier_inference(
        self,
        data_loader: DataLoader,
        msg: str = "Acc Eval",
        progress: bool = True,
        verbose=True,
    ):
        self.net.eval()

        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=msg, disable=not progress or not verbose):
                data = batch["data"].cuda()
                logits = self.net(data)
                preds = logits.argmax(1)
                all_preds.append(preds.cpu())
                all_labels.append(batch["label"])

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        return all_preds, all_labels

    def eval_acc(self, data_name: str = "id", verbose: bool = True) -> float:
        """
        Evaluate classification accuracy for ID or CSID.
        """
        if data_name == "id":
            if self.metrics["id_acc"] is not None:
                return self.metrics["id_acc"]
            else:
                if self.scores["id_preds"] is None:
                    all_preds, all_labels = self._classifier_inference(
                        self.dataloader_dict["id"]["test"],
                        "ID Acc Eval",
                        progress=verbose,
                    )
                    self.scores["id_preds"] = all_preds
                    self.scores["id_labels"] = all_labels
                else:
                    all_preds = self.scores["id_preds"]
                    all_labels = self.scores["id_labels"]

                assert len(all_preds) == len(all_labels)
                correct = (all_preds == all_labels).sum().item()
                acc = correct / len(all_labels) * 100
                self.metrics["id_acc"] = acc
                return acc
        elif data_name == "csid":
            if self.metrics["csid_acc"] is not None:
                return self.metrics["csid_acc"]
            else:
                correct, total = 0, 0
                for _, (dataname, dataloader) in enumerate(self.dataloader_dict["csid"].items()):
                    if self.scores["csid_preds"][dataname] is None:
                        all_preds, all_labels = self._classifier_inference(
                            dataloader,
                            f"CSID {dataname} Acc Eval",
                            progress=verbose,
                        )
                        self.scores["csid_preds"][dataname] = all_preds
                        self.scores["csid_labels"][dataname] = all_labels
                    else:
                        all_preds = self.scores["csid_preds"][dataname]
                        all_labels = self.scores["csid_labels"][dataname]

                    assert len(all_preds) == len(all_labels)
                    c = (all_preds == all_labels).sum().item()
                    t = len(all_labels)
                    correct += c
                    total += t

                # also combine with id test if you want
                if self.scores["id_preds"] is None:
                    all_preds, all_labels = self._classifier_inference(
                        self.dataloader_dict["id"]["test"],
                        "ID Acc Eval",
                        progress=verbose,
                    )
                    self.scores["id_preds"] = all_preds
                    self.scores["id_labels"] = all_labels
                else:
                    all_preds = self.scores["id_preds"]
                    all_labels = self.scores["id_labels"]

                correct += (all_preds == all_labels).sum().item()
                total += len(all_labels)

                acc = correct / total * 100
                self.metrics["csid_acc"] = acc
                return acc
        else:
            raise ValueError(f"Unknown data name {data_name}")

    def eval_ood(self, fsood: bool = False, progress: bool = True, verbose: bool = True):
        """
        Evaluate OOD detection performance, but ONLY for near OOD.

        i.e.:
        - id_name = "cifar10"
        - 'near' OOD = "cifar100" (and possibly other near sets, if specified)
        - We skip any 'far' references entirely.
        """
        id_name = "id" if not fsood else "csid"
        task = "ood" if not fsood else "fsood"
        if self.metrics[task] is None:
            self.net.eval()

            # --- ID Scores
            if self.scores["id"]["test"] is None:
                if verbose:
                    print(f"Performing inference on {self.id_name} test set...", flush=True)
                id_pred, id_conf, id_gt = self.postprocessor.inference(
                    self.net, self.dataloader_dict["id"]["test"], progress and verbose
                )
                self.scores["id"]["test"] = [id_pred, id_conf, id_gt]
            else:
                id_pred, id_conf, id_gt = self.scores["id"]["test"]

            # If FSOOD, combine ID and csid
            if fsood:
                csid_pred, csid_conf, csid_gt = [], [], []
                for i, dataset_name in enumerate(self.scores["csid"].keys()):
                    if self.scores["csid"][dataset_name] is None:
                        if verbose:
                            print(
                                f"Performing inference on {self.id_name} "
                                f"(cs) test set [{i+1}]: {dataset_name}...",
                                flush=True,
                            )
                        temp_pred, temp_conf, temp_gt = self.postprocessor.inference(
                            self.net,
                            self.dataloader_dict["csid"][dataset_name],
                            progress and verbose,
                        )
                        self.scores["csid"][dataset_name] = [temp_pred, temp_conf, temp_gt]

                    csid_pred.append(self.scores["csid"][dataset_name][0])
                    csid_conf.append(self.scores["csid"][dataset_name][1])
                    csid_gt.append(self.scores["csid"][dataset_name][2])

                csid_pred = np.concatenate(csid_pred)
                csid_conf = np.concatenate(csid_conf)
                csid_gt = np.concatenate(csid_gt)

                id_pred = np.concatenate((id_pred, csid_pred))
                id_conf = np.concatenate((id_conf, csid_conf))
                id_gt = np.concatenate((id_gt, csid_gt))

            # --- NEAR OOD
            near_metrics = self._eval_ood(
                [id_pred, id_conf, id_gt],
                ood_split="near",
                progress=progress and verbose,
                verbose=verbose,
            )

            # skip 'far' entirely

            # attach ID ACC if available
            if self.metrics[f"{id_name}_acc"] is None:
                self.eval_acc(id_name, verbose=verbose)

            near_metrics[:, -1] = np.array([self.metrics[f"{id_name}_acc"]] * len(near_metrics))

            # combine near metrics into final
            self.metrics[task] = pd.DataFrame(
                near_metrics,
                index=list(self.dataloader_dict["ood"]["near"].keys()) + ["nearood"],
                columns=["FPR@95", "AUROC", "AUPR_IN", "AUPR_OUT", "ACC"],
            )
        else:
            if verbose:
                print("Evaluation has already been done!")

        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.float_format",
            "{:,.2f}".format,
        ):
            if verbose:
                print(self.metrics[task])

        return self.metrics[task]

    def _eval_ood(
        self,
        id_list: List[np.ndarray],
        ood_split: str = "near",
        progress: bool = True,
        verbose: bool = True,
    ):
        """
        Evaluate OOD detection for the specified ood_split, here only "near".
        """
        if verbose:
            print(f"Processing {ood_split} ood...", flush=True)
        [id_pred, id_conf, id_gt] = id_list
        metrics_list = []
        for dataset_name, ood_dl in self.dataloader_dict["ood"][ood_split].items():
            if self.scores["ood"][ood_split][dataset_name] is None:
                if verbose:
                    print(f"Performing inference on {dataset_name} dataset...", flush=True)
                ood_pred, ood_conf, ood_gt = self.postprocessor.inference(
                    self.net, ood_dl, progress and verbose
                )
                self.scores["ood"][ood_split][dataset_name] = [ood_pred, ood_conf, ood_gt]
            else:
                if verbose:
                    print(
                        f"Inference has been performed on {dataset_name} dataset...",
                        flush=True,
                    )
                [ood_pred, ood_conf, ood_gt] = self.scores["ood"][ood_split][dataset_name]

            # label OOD samples as -1
            ood_gt = -1 * np.ones_like(ood_gt)
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])

            if verbose:
                print(f"Computing metrics on {dataset_name} dataset...")
            ood_metrics = compute_all_metrics(conf, label, pred)
            metrics_list.append(ood_metrics)
            if verbose:
                self._print_metrics(ood_metrics)

        if verbose:
            print("Computing mean metrics...", flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0, keepdims=True)
        if verbose:
            self._print_metrics(list(metrics_mean[0]))

        # append the average row
        combined = np.concatenate([metrics_list, metrics_mean], axis=0) * 100
        return combined

    def _print_metrics(self, metrics):
        [fpr, auroc, aupr_in, aupr_out, _] = metrics

        # print ood metric results
        print(
            "FPR@95: {:.2f}, AUROC: {:.2f}".format(100 * fpr, 100 * auroc),
            end=" ",
            flush=True,
        )
        print(
            "AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}".format(100 * aupr_in, 100 * aupr_out),
            flush=True,
        )
        print("\u2500" * 70, flush=True)
        print("", flush=True)

    def hyperparam_search(self, verbose: bool = False):
        if verbose:
            print("Starting automatic parameter search...")
        max_auroc = 0
        hyperparam_names = []
        hyperparam_list = []
        count = 0

        for name in self.postprocessor.args_dict.keys():
            hyperparam_names.append(name)
            count += 1

        for name in hyperparam_names:
            hyperparam_list.append(self.postprocessor.args_dict[name])

        hyperparam_combination = self.recursive_generator(hyperparam_list, count)

        final_index = None
        for i, hyperparam in enumerate(
            tqdm(hyperparam_combination, disable=not verbose, desc="Hyperparam Search")
        ):
            self.postprocessor.set_hyperparam(hyperparam)
            if self.postprocessor_name == "weiper_kldiv":
                if i == 0:
                    old_hyperparam = hyperparam[-2]
                    self.postprocessor.setup(
                        self.net,
                        self.dataloader_dict["id"],
                        None,
                        hyperparamter_search=True,
                    )
                else:
                    if hyperparam[-2] != old_hyperparam:
                        # if n_bins or perturbation_distance changes, we need to re-setup
                        self.postprocessor.setup(
                            self.net,
                            self.dataloader_dict["id"],
                            None,
                            hyperparamter_search=True,
                        )
                        old_hyperparam = hyperparam[-2]

            id_pred, id_conf, id_gt = self.postprocessor.inference(
                self.net, self.dataloader_dict["id"]["val"], progress=False
            )
            ood_pred, ood_conf, ood_gt = self.postprocessor.inference(
                self.net, self.dataloader_dict["ood"]["val"], progress=False
            )

            ood_gt = -1 * np.ones_like(ood_gt)
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_all_metrics(conf, label, pred)
            auroc = ood_metrics[1]

            if auroc > max_auroc:
                final_index = i
                max_auroc = auroc

        self.postprocessor.set_hyperparam(hyperparam_combination[final_index])
        if self.postprocessor_name == "weiper_kldiv":
            self.postprocessor.setup(
                self.net,
                self.dataloader_dict["id"],
                None,
                hyperparamter_search=False,
            )
        if verbose:
            print("Final hyperparam: {}".format(self.postprocessor.get_hyperparam()))
        self.postprocessor.hyperparam_search_done = True

    def recursive_generator(self, list_, n):
        if n == 1:
            results = []
            for x in list_[0]:
                k = []
                k.append(x)
                results.append(k)
            return results
        else:
            results = []
            temp = self.recursive_generator(list_, n - 1)
            for x in list_[n - 1]:
                for y in temp:
                    k = y.copy()
                    k.append(x)
                    results.append(k)
            return results
