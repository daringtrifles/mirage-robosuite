import argparse
import json

import numpy as np

import robosuite as suite
import robosuite.utils.transform_utils as T
import robosuite.utils.camera_utils as camera_utils
from robosuite.controllers import load_controller_config
from robosuite.renderers import load_renderer_config
from robosuite.utils.input_utils import *
import robosuite.macros as macros
macros.IMAGE_CONVENTION = "opencv"

def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":

    """
    Registered environments: Lift, Stack, NutAssembly, NutAssemblySingle, NutAssemblySquare, NutAssemblyRound,
                             PickPlace, PickPlaceSingle, PickPlaceMilk, PickPlaceBread, PickPlaceCereal,
                             PickPlaceCan, Door, Wipe, TwoArmLift, TwoArmPegInHole, TwoArmHandover

    Possible robots: Baxter, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e
    """

    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--renderer", type=str, default="mujoco", help="Valid options include mujoco, and nvisii")

    args = parser.parse_args()
    renderer = args.renderer

    options["env_name"] = choose_environment()

    # If a multi-arm environment has been chosen, choose configuration and appropriate robot(s)
    if "TwoArm" in options["env_name"]:
        # Choose env config and add it to options
        options["env_configuration"] = choose_multi_arm_config()

        # If chosen configuration was bimanual, the corresponding robot must be Baxter. Else, have user choose robots
        if options["env_configuration"] == "bimanual":
            options["robots"] = "Baxter"
        else:
            options["robots"] = []

            # Have user choose two robots
            print("A multiple single-arm configuration was chosen.\n")

            for i in range(2):
                print("Please choose Robot {}...\n".format(i))
                options["robots"].append(choose_robots(exclude_bimanual=True))

    # Else, we simply choose a single (single-armed) robot to instantiate in the environment
    else:
        options["robots"] = choose_robots(exclude_bimanual=True)

    # Choose controller
    controller_name = choose_controller()

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)

    env = suite.make(
        **options,
        has_renderer=False if renderer != "mujoco" else True,  # no on-screen renderer
        has_offscreen_renderer=True,  # no off-screen renderer
        ignore_done=True,
        use_camera_obs=True,  # no camera observations
        control_freq=20,
        renderer=renderer,
        camera_names = ["frontview", "sideview"],
        camera_depths = True,
        camera_segmentations = "robot_only",
    )

    env.reset()

    low, high = env.action_spec

    if renderer == "nvisii":

        timesteps = 300
        for i in range(timesteps):
            action = np.random.uniform(low, high)
            obs, reward, done, _ = env.step(action)

            if i % 100 == 0:
                env.render()

    else:

        # do visualization
        for i in range(10000):
            action = np.random.uniform(low, high)
            obs, reward, done, _ = env.step(action)
            # env.render()
            breakpoint()
            import matplotlib.pyplot as plt
            # plt.imshow(obs['frontview_image']); plt.show()
            # plt.imshow(obs['frontview_depth']); plt.show() # (256, 256, 1)
            depth_map = obs['frontview_depth']
            # plt.imshow(obs['frontview_segmentation_robot_only']); plt.show() # (256, 256, 1)
            real_depth_map = camera_utils.get_real_depth_map(env.sim, depth_map)
            val = env.sim.render(height=512, width=512, camera_name="sideview")
            plt.imshow(val); plt.show()

    env.close_renderer()
    print("Done.")
