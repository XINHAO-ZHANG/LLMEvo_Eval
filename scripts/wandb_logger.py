import os
import json
import wandb
from typing import Any

import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA

def _hash_vec(obj: Any, dim: int = 16) -> np.ndarray:
    h = abs(hash(json.dumps(obj, sort_keys=True)))
    vec = np.zeros(dim)
    for i in range(dim):
        vec[i] = (h >> (i * 4)) & 0xF    # 4-bit chunks
    return vec
def make_wandb_callback(project: str, cfg: dict):
    run = wandb.init(project=project, config=cfg)

    def _cb(log):
        # core fitness stats
        scores = log["child_scores"]
        run.log({
            "best_so_far":  log["best_so_far"],
            "child_score/min":    min(scores),
            "child_score/mean":   sum(scores)/len(scores),
            "child_score/std":    (pd.Series(scores).std()),
        },step=log["gen"])

        # ---------------- diversity on population ----------------
        pop = log.get("population", [])
        if pop:
            pop_scores = [p["score"] for p in pop]
            div_keys   = [hash(json.dumps(p["genome"], sort_keys=True)) for p in pop]
            p_series   = pd.Series(div_keys).value_counts(normalize=True)
            entropy    = -(p_series * np.log(p_series)).sum()
            run.log({
                "diversity/unique":    int(p_series.size),
                "diversity/shannon_H": float(entropy),
                "population/size":     len(pop),
                "population/mean":     float(np.mean(pop_scores)),
            }, step=log["gen"])

            # # Scatter: embed genomes to 2-d with PCA on hash vectors
            # vecs = np.vstack([_hash_vec(p["genome"]) for p in pop])
            # coords = PCA(n_components=2, random_state=0).fit_transform(vecs)

            # # prepare wandb Table
            # table = wandb.Table(columns=["x", "y", "score"])
            # for (x, y), p in zip(coords, pop):
            #     table.add_data(float(x), float(y), float(p["score"]))
            # gen_idx = log["gen"]
            # run.log({
            #     "diversity/scatter": wandb.plot.scatter(
            #         table, "x", "y",
            #         title=f"Pop PCA scatter (gen {gen_idx})")
            # }, step=gen_idx)

        # save all population and children to CSV
        gen_idx = log["gen"]
        pop = log.get("population", [])
        parent_rows = [
            {"generation": gen_idx, "genome": json.dumps(p["genome"]), "score": p["score"], "type": "parent"}
            for p in pop
        ]

        children = log.get("children", [])
        child_scores = log.get("child_scores", [])
        child_rows = [
            {"generation": gen_idx, "genome": json.dumps(g), "score": s, "type": "child"}
            for g, s in zip(children, child_scores)
        ]

        all_rows = parent_rows + child_rows
        df = pd.DataFrame(all_rows)

        model_name = cfg.get("model", "unknownmodel")
        safe_model_name = model_name.replace("/", "_")
        n_parent = cfg.get("parent_slots", "0")
        n_child = cfg.get("child_slots", "0")
        db_mode = cfg.get("db_mode", "unknowndb")
        csv_path = f"{safe_model_name}_np{n_parent}_nc{n_child}_{db_mode}.csv"   
        write_header = not os.path.exists(csv_path)
        df.to_csv(csv_path, mode="a", header=write_header, index=False)

        # upload CSV to wandb
        run.save(csv_path)

    return _cb