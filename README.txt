#NO copy walker2d.xml to ./venv/lib/python3.10/site-packages/gymnasium/envs/mujoco/assets/walker2d.xml
# fixes a bug in joint ranges
use danx0r forked Gymnasium -- it has these fixes + stompy

gymnasitc is our repo

Stompy Joints:
Assume robot facing towards X+ axis
Z = transverse
Y = Sagittal
X = Frontal

hip:
    left:
        Z: joint_legs_1_left_leg_1_x8_1_dof_x8 (works)
        1 = counterclockwise, -1 = clockwise (looking down from above)
        Y: joint_legs_1_left_leg_1_x10_1_dof_x10
        1 = extended (backbend), -1 = contracted (Knee to nose)
    right:
        Z: joint_legs_1_right_leg_1_x8_1_dof_x8 (works)
        1 = counterclockwise, -1 = clockwise (looking down from above)
        Y: joint_legs_1_right_leg_1_x10_2_dof_x10 (works)
        1 = contracted (knee to nose), -1 = extended (backbend/shark pose)

knee:
    left:
        joint_legs_1_left_leg_1_knee_revolute
        1 = extended, -1 = contracted
    right:
        joint_legs_1_right_leg_1_knee_revolute
        1 = contracted; -1 = extended

ankle:
    left:
        joint_legs_1_left_leg_1_ankle_revolute
        1 = extended (high heels/ballet); -1 contracted (walk on heels)

    lright:
        joint_legs_1_left_leg_1_ankle_revolute
        1 = contracted (walk on heels); -1 = extended (tippytoe)

foot:
    left:
        joint_legs_1_left_leg_1_x4_1_dof_x4
        1 = extended (foot rolled outward, away from centerline)
        -1 = contracted (foot rolled towards instep)
    right:
        joint_legs_1_right_leg_1_x4_1_dof_x4
        1 = extended (foot rolled outward, away from centerline)
        -1 = contracted (foot rolled towards instep)