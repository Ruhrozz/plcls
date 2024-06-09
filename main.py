import argparse

from omegaconf import OmegaConf

from plcls.run import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file")
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)
    train(conf)
