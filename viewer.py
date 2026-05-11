import os

import mujoco
import mujoco.viewer


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_PATH = os.path.join(ROOT_DIR, "Scene.xml")


def main() -> None:
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
