"""
Run all scripts necessary to reproduce the paper.
"""

import sys
from dataclasses import dataclass

sys.path.append("modules")

import setup_parameters
import setup_OFC_network


@dataclass
class Options:
    run_id: int = 0 # random seed
    out_dir: str # path to save the results
    dataset_name: str = "Reaching"
    task: str = "random" # options: ["random", "pushed"]
    fb_density: int = 0
    fb_in: int = 0 # options: [0, 1]
    delay: int = 0 # options: [0, 1]
    learning_rate: float = 1e-3 
    rot_phi: int = 30 # perturbation angle
    vel: int = 10  # velocity
    batch_size: int = 256 
    go_to_peak: int = 50 # velocity curve shape
    custom_delay: dict = None # delay range for the start of movement, see below
    fb_freeze: bool = False # freeze feedback initially
    error_detach: bool = False # detach error signal from the computational graph before integration
    get_grads_per_example: bool = False 
    wfb_frozen_phase: int = 0 # phase of the feedback loop where the weights are frozen
    pratio: float = 0.25 # ratio of the perturbed vs non-perturbed trials


def run(opts):

    # this effectively removes the delay in start of movement
    custom_delay = {}
    custom_delay["r_go_range"] = [70, 71]
    custom_delay["cor_go_range"] = [70, 71]

    savname = opts.out_dir
    setup_parameters.main(
        savname=savname,
        dataset_name=opts.dataset_name,
        rand_seed=opts.run_id,
        fb_density=opts.fb_density,
        fb_delay=opts.delay,
        protocol=[[opts.task, 5000]],
        rot_phi=opts.rot_phi,
        learning_rate=opts.learning_rate,
        vel=opts.vel,
        batch_size=opts.batch_size,
        go_to_peak=opts.go_to_peak,
        custom_delay=custom_delay,
        fb_freeze=opts.fb_freeze,
        error_detach=opts.error_detach,
        get_grads_per_example=opts.get_grads_per_example,
        wfb_frozen_phase=opts.wfb_frozen_phase,
        pratio=opts.pratio,
    )
    setup_OFC_network.main(savname)


def main():

    import simple_parsing
    from simple_parsing import DashVariant

    config = simple_parsing.parse(Options, add_option_string_dash_variant=DashVariant.DASH)

    run(config)


if __name__ == "__main__":
    main()