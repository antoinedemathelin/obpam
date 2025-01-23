import model_wrappers as mw
import data_loaders as dl
import run
import argparse
import gc
import numpy as np
import os
import yaml
import time


methods_dict = {
    "onebatch": mw.one_batch,
    "onebatch-nniw": mw.one_batch_nniw,
    "onebatch-lwcs": mw.one_batch_lwcs,
    "onebatch-bias": mw.one_batch_bias,
    "banditpam": mw.banditpam,
    "random": mw.random_model,
    "greedy": mw.greedy,
    "pam": mw.pam,
    "kmeans++": mw.kmeans_pp,
    "kmeans++ls": mw.kmeans_pp_ls,
    "clara": mw.clara,
    "fastclara": mw.fast_clara,
    "kmc2": mw.kmc2
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Running experiment")
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    # default config
    config = dict(
        save_dir="results",
        datasets=["mnist"],
        methods=["onebatch"],
        N=None,
        K=[10],
        seeds=[0],
        params={},
        metric="l1",
        timeout=600
    )
    
    # open config
    file = open("configs/%s.yml"%args.config, "r")
    config.update(yaml.safe_load(file))
    file.close()

    for k, v in config.items():
        print(k+": ", v)
    
    os.makedirs(os.path.join(".", config["save_dir"]), exist_ok=True)
    
    for dataset in config["datasets"]:

        out_name = os.path.join(config["save_dir"], dataset + ".csv")

        methods_params_dict = []
        for method in config["methods"]:
            if method in config["params"]:
                param_name = list(config["params"][method].keys())[0]
                methods_params_dict += [(methods_dict[method], {param_name: p})
                                        for p in config["params"][method][param_name]]
            else:
                methods_params_dict += [(methods_dict[method], dict())]
        
        for method, params in methods_params_dict:
            try:
                run.run(dataset=dataset,
                       seeds=config["seeds"],
                       K_list=config["K"],
                       sample_sizes=config["N"],
                       metric=config["metric"],
                       method=method,
                       out_name=out_name,
                       timeout=config["timeout"],
                       save=True,
                       params=params)
                gc.collect()
            except:
                gc.collect()
                print("Error")
                time.sleep(3)