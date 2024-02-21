from ikomia.dataprocess.workflow import Workflow
from ikomia.utils import ik
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add the MMpose algorithm
pose = wf.add_task(ik.infer_mmlab_pose_estimation(), auto_connect=True)

# Run directly on your image
wf.run_on(https://github.com/Kygosaur/Portfolio/blob/e7f284bb5f81a4ab47764d04e897de4f3a1ad478/datasets/coco8-pose/images/train/000000000036.jpg)

# Inspect your result
display(pose.get_image_with_graphics())