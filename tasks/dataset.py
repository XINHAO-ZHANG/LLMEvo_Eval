from datasets import load_dataset


# --------------------------------------------------------------------------- #
# PromptOpt dataset
# --------------------------------------------------------------------------- #
def get_samsum_dataset(split="test"):
    ds = load_dataset("samsum", split=split)
    # 返回 [{'dialogue': ..., 'summary': ...}, ...]
    return [{"dialogue": x["dialogue"], "summary": x["summary"]} for x in ds]

def get_asset_dataset(split="test"):
    ds = load_dataset("asset", split=split)
    # 返回 [{'source': ..., 'references': [...]}, ...]
    return [{"source": x["source"], "references": x["references"]} for x in ds]