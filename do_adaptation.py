"""
Run all scripts necessary to reproduce the paper.
"""

import sys
from dataclasses import dataclass

sys.path.append("modules")

import adaptation_learning


@dataclass
class Options:
    run_id: int = 0 # random seed
    out_dir: str # path to save the results
    dataset_name: str = "Reaching" # options: ["Reaching", "Sinewave"]
    task: str = "random" # options: ["random", "pushed"]
    fb_density: int = 0 # options: [0, 1]
    fb_in: int = 0 # options: [0, 1]
    tm_savname: str # path to the pretrained model checkpoint
    learning_algorithm: str # options: ["rflo", "fed", "bp"]
    ad_learning_rate: float # learning rate for rflo/fed
    bp_learning_rate: float # learning rate for backpropagation
    bp_opt: str = "adam" # options: ["adam", "sgd"]
    ntrials: int = 500 # number of trials
    batch_size: int = 1 # batch size for rflo/fed
    rot_phi: int = 30 # rotation angle for the input
    

def run(opts):
    savname = opts.out_dir

    if opts.learning_algorithm == "bp":
        if opts.fb_density != 0:
            learning_rate = opts.bp_learning_rate
        else:
            learning_rate = opts.bp_learning_rate * 1 / 10
    else:
        if opts.fb_density != 0:
            if opts.fb_in == 0:
                learning_rate = opts.ad_learning_rate * 1 / 10
            else:
                learning_rate = opts.ad_learning_rate
        else:
            learning_rate = opts.ad_learning_rate * 1 / 10

    if opts.learning_algorithm not in ["rflo", "fed"]:
        opts.record_local_gradients = False

    if opts.fb_density == 0 and opts.learning_algorithm in ["rflo", "fed"]:
        opts.learning_algorithm += "_t"  # use the transpose!

    adaptation_learning.main(
        savname,
        opts.learning_algorithm,
        learning_rate,
        opts.fb_in,
        bp_opt=opts.bp_opt,
        tm_savname=opts.tm_savname,
        ntrials=opts.ntrials,
        batch_size=opts.batch_size,
        rot_phi=opts.rot_phi,
    )


def main():

    import simple_parsing
    from simple_parsing import DashVariant

    config = simple_parsing.parse(Options, add_option_string_dash_variant=DashVariant.DASH)

    run(config)

if __name__ == "__main__":
    main()