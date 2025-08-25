import os
from omegaconf import OmegaConf

def get_config(yml_file="configs/profile_baseline.yaml"):
    if not os.path.exists(yml_file):
        print(f"The config file passed {yml_file} doesn't exist.")
        raise ValueError
    yml_conf = OmegaConf.load(yml_file)
    cli_conf = OmegaConf.from_cli()
    conf = OmegaConf.merge(yml_conf, cli_conf)
    if not conf.do_sample:
        conf.top_k = None
        conf.temperature = None
    return conf