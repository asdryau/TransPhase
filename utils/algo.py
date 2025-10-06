from pathlib import Path
import torch
import numpy as np
from utils.rotation_conversion import *
from smplx import SMPL

####
#   helper function
###
smpl = SMPL(
        model_path=Path("./utils/SMPL_FEMALE.pkl"),
        gender='FEMALE',
        batch_size=1,
        dtype=torch.float)


def save_motion(save_filename, root_traj, local_motion):
    """
    Args:
        root_traj: (T, 3)
        local_motion: (T, J*3)
    Returns:
        None
    """
    T,E = local_motion.shape
    root_traj = root_traj.detach().cpu().numpy()
    local_motion = local_motion.detach().cpu().numpy()
    # print("local_motion", local_motion.shape)
    local_motion = np.pad(local_motion, ((0, 0), (0, 55*3-E)))
    # print("local_motionpad", local_motion.shape)

    np.savez(save_filename,
        poses=local_motion.reshape((T, -1)), trans=root_traj, betas=np.zeros(10), gender="female", mocap_framerate=30, dmpls=np.zeros((T,8)))

def forward_kinematics(root_traj, local_motion):
    """
    Args:
        root_traj: (B, 3)
        local_motion: (B, J, 3, 3)
    Returns:
        joint_pos : (B, J, 3)
    """
    global smpl
    smpl = smpl.to(local_motion.device)
    B,J = local_motion.shape[:2]
    if len(local_motion.shape) == 4:
        local_motion = matrix_to_axis_angle(local_motion)[:,:24]    # 24 joints
    else:
        local_motion = local_motion[:,:24]
    local_motion = torch.nn.functional.pad(local_motion, (0, 0, 0, 24-J))    # 24 joints
    root_rot, poses = local_motion[:,0], local_motion[:, 1:].reshape((local_motion.shape[0], -1))
    generated_smpl = smpl.forward(body_pose=poses, global_orient=root_rot, transl=root_traj, return_verts=False)
    joint_pos = generated_smpl.joints
    return joint_pos[:,:22]

def forward_kinematics_gpu(smpl_model, root_traj, local_motion):
    """
    Args:
        root_traj: (B, 3)
        local_motion: (B, J, 3, 3)
    Returns:
        joint_pos : (B, J, 3)
    """
    smpl_model = smpl_model.to(local_motion.device)
    B,J = local_motion.shape[:2]
    if len(local_motion.shape) == 4:
        local_motion = matrix_to_axis_angle(local_motion)[:,:24]    # 24 joints
    else:
        local_motion = local_motion[:,:24]
    local_motion = torch.nn.functional.pad(local_motion, (0, 0, 0, 24-J))    # 24 joints
    root_rot, poses = local_motion[:,0], local_motion[:, 1:].reshape((local_motion.shape[0], -1))
    generated_smpl = smpl_model.forward(body_pose=poses, global_orient=root_rot, transl=root_traj, return_verts=False)
    joint_pos = generated_smpl.joints
    return joint_pos[:,:22]

########
#   data preprocessing function
#######
def motion_preprocess(trans, poses):
    # trans : (T,3)
    T = poses.shape[0]
    poses = poses.reshape((T, -1,3)) #(T,J,3)
    poses = axis_angle_to_quaternion(poses) #(T,J,4)

    # print("trans, poses", trans.shape, poses.shape) # torch.Size([47, 3]) torch.Size([47, 24, 4])
    ###########
    # we follow the step from HumanML3D but with some modifications, also we use smplx package and pytorch3D rotation_conversion

    ### 1) FK
    global_positions = forward_kinematics(trans, quaternion_to_matrix(poses)) #(T,J,3)
    ### 2) put on floor (AMASS is Z-up)
    floor_height = global_positions.min(axis=0)[0].min(axis=0)[0][2]
    global_positions[:,:,2] -= floor_height

    # ### 3) set initital frame XY at (0,0)
    global_positions[:,:,0] = global_positions[:,:,0] - global_positions[0:1,0:1,0]      # Note that height (z position) is preserved
    global_positions[:,:,1] = global_positions[:,:,1] - global_positions[0:1,0:1,1]      # Note that height (z position) is preserved

    ### 4) initial frame face Y+
    last_pose = matrix_to_quaternion(torch.eye(3)).to(poses.device)    #(4,)   # define y+ forward orientation
    next_pose = poses[0, 0].clone()               #(4,)   # this is the root orientation of center frame (you can change it to frame0, frameT, etc...)
    root_rotate_start = quaternion_multiply(last_pose, quaternion_invert(next_pose))  # (4,)
        # normalize root_rotate_start to only work on horizontal orientation, not elevation
            ### for AMASS format (Z-up)
    root_rotate_startY = matrix_to_euler_angles(quaternion_to_matrix(root_rotate_start), "XYZ")
    root_rotate_startY[0] = 0.
    root_rotate_startY[1] = 0.
    root_rotate_start = matrix_to_quaternion(euler_angles_to_matrix(root_rotate_startY, "XYZ"))
    poses[:,0] = quaternion_multiply(root_rotate_start.unsqueeze(0), poses[:,0])
    #     ### also rotate global_positions
    global_positions = quaternion_apply(root_rotate_start.unsqueeze(0), global_positions)

    # print("global_positions", global_positions.shape)     #torch.Size([47, 22, 3])

    ### 5) get joint rotation in 6D
    poses_6d = matrix_to_rotation_6d(quaternion_to_matrix(poses))
    # print("poses_6d", poses_6d.shape)   # torch.Size([47, 24, 6])

    ### 6) get root representation !!!!!!
    root_pose = poses[:,0]
    trans = global_positions[:,0]
        # calc root_rotate that set all frame facing Y+
    last_pose = matrix_to_quaternion(torch.eye(3)).to(poses.device)    #(4,)   # define y+ forward orientation
    root_rotate_all = quaternion_multiply(last_pose.unsqueeze(0), quaternion_invert(root_pose))  # (T,4)
    root_rotate_allY = matrix_to_euler_angles(quaternion_to_matrix(root_rotate_all), "XYZ")
    root_rotate_allY[:,0] = 0.
    root_rotate_allY[:,1] = 0.
    root_rotate_all = matrix_to_quaternion(euler_angles_to_matrix(root_rotate_allY, "XYZ"))
        # calc linear velocity
    traj_vel = trans[1:] - trans[:-1]
    traj_vel = quaternion_apply(root_rotate_all[:-1], traj_vel)
        # calc root angular velocity
    root_rotvelZ =  quaternion_multiply(root_pose[1:], quaternion_invert(root_pose[:-1]))
    root_rotvelZ = matrix_to_euler_angles(quaternion_to_matrix(root_rotvelZ), "XYZ")[:,2]
    # print("traj_vel, root_rotvelZ", traj_vel.shape, root_rotvelZ.shape)   # torch.Size([46, 3]) torch.Size([46])

    ### 7) get rifke
    root_rot_Zfiltered = quaternion_multiply(root_rotate_all, root_pose)
        # FK again to obtain rifke
    pose_modified = poses.clone()
    pose_modified[:,0] = root_rot_Zfiltered
    trans_modified = global_positions[:,0].clone()
    trans_modified[:,0] = 0.
    trans_modified[:,1] = 0.
    rifke = forward_kinematics(trans_modified, quaternion_to_matrix(pose_modified)) #(T,J,3)
    # print("root_rot_Zfiltered, rifke", root_rot_Zfiltered.shape, rifke.shape) # torch.Size([47, 4]) torch.Size([47, 22, 3])

    ### 8) get local velocity
    local_vel = global_positions[1:] - global_positions[:-1]
    local_vel = quaternion_apply(root_rotate_all[:-1].unsqueeze(1), local_vel)
    # print("local_vel", local_vel.shape)   # torch.Size([46, 22, 3])

    ### 9) foot contact
    feet_l = (torch.norm(global_positions[1:, [7, 10]] - global_positions[:-1, [7, 10]], dim=2) < 0.002).float()
    feet_r = (torch.norm(global_positions[1:, [8, 11]] - global_positions[:-1, [8, 11]], dim=2) < 0.002).float()
    # print("feet_l, feet_r", feet_l.shape, feet_r.shape)   # torch.Size([46, 2]) torch.Size([46, 2])

    ### concatnate data
    data = torch.cat([root_rotvelZ.unsqueeze(-1), matrix_to_rotation_6d(quaternion_to_matrix(root_rot_Zfiltered[:-1])),\
                        traj_vel[:,0:2], trans[:-1,2:3],\
                        poses_6d[:-1,1:22].reshape(T-1,-1), rifke[:-1,1:22].reshape(T-1,-1), local_vel[:,:22].reshape(T-1,-1), feet_l, feet_r], dim=-1)
    # print("data", data.shape)   #torch.Size([46, 269])
    return data


def motion_to_smpl(motion):
    ### root traj
    root_rotvelZ, root_rot_Zfiltered = motion[:,0], motion[:,1:7]
    traj_vel, transZ = motion[:,7:9], motion[:,9:10]
        # process rotation velocity
    root_rotvelZ = torch.cat([root_rotvelZ[-1:]*0., root_rotvelZ[:-1]], dim=0)  # shift-1 in dim 0
    root_rotZ = torch.cumsum(root_rotvelZ, dim=0)
    root_rotZ = torch.stack([root_rotZ*0., root_rotZ*0., root_rotZ],dim=-1)
    root_rotZ = matrix_to_quaternion(euler_angles_to_matrix(root_rotZ, "XYZ"))

        # reconstruct root_rot
    root_rot_Zfiltered = matrix_to_quaternion(rotation_6d_to_matrix(root_rot_Zfiltered))
    root_rot = quaternion_multiply(root_rotZ, root_rot_Zfiltered)   #(T,4)
    root_rot = matrix_to_axis_angle(quaternion_to_matrix(root_rot))
        # traj
    traj_vel = torch.cat([traj_vel[-1:]*0., traj_vel[:-1]], dim=0)              # shift-1 in dim 0
    traj_vel = torch.cat([traj_vel, traj_vel[:, 0:1]*0.], dim=1)
    traj_vel = quaternion_apply(root_rotZ, traj_vel)[:,0:2]
    transXY = torch.cumsum(traj_vel, dim=0)
    trans = torch.cat([transXY, transZ], dim=-1)                    #(T,3)


    ### poses
    poses_6d = motion[:,10:136].reshape((-1,21,6))
    poses = matrix_to_axis_angle(rotation_6d_to_matrix(poses_6d))   #(T,21,3)
    # print("root_rot, poses, trans", root_rot.shape, poses.shape, trans.shape)   # torch.Size([46, 6]) torch.Size([46, 21, 3]) torch.Size([46, 3])
    poses = torch.cat([root_rot.unsqueeze(1), poses], dim=1).reshape((-1,22*3))       #(T,22,3)

    return trans, poses

def motion_to_smpl_batch(motion):
    ### root traj
    root_rotvelZ, root_rot_Zfiltered = motion[:,:,0], motion[:,:,1:7]
    traj_vel, transZ = motion[:,:,7:9], motion[:,:,9:10]
        # process rotation velocity
    root_rotvelZ = torch.cat([root_rotvelZ[:,-1:]*0., root_rotvelZ[:,:-1]], dim=1)  # shift-1 in dim 1
    root_rotZ = torch.cumsum(root_rotvelZ, dim=1)
    root_rotZ = torch.stack([root_rotZ*0., root_rotZ*0., root_rotZ],dim=-1)
    root_rotZ = matrix_to_quaternion(euler_angles_to_matrix(root_rotZ, "XYZ"))

        # reconstruct root_rot
    root_rot_Zfiltered = matrix_to_quaternion(rotation_6d_to_matrix(root_rot_Zfiltered))
    root_rot = quaternion_multiply(root_rotZ, root_rot_Zfiltered)   #(T,4)
    root_rot = matrix_to_axis_angle(quaternion_to_matrix(root_rot))
        # traj
    traj_vel = torch.cat([traj_vel[:,-1:]*0., traj_vel[:,:-1]], dim=1)              # shift-1 in dim 1
    traj_vel = torch.cat([traj_vel, traj_vel[:,:,0:1]*0.], dim=2)
    traj_vel = quaternion_apply(root_rotZ, traj_vel)[:,:,0:2]
    transXY = torch.cumsum(traj_vel, dim=1)
    trans = torch.cat([transXY, transZ], dim=-1)                    #(T,3)


    ### poses
    poses_6d = motion[:,:,10:136].reshape((motion.shape[0],-1,21,6))
    poses = matrix_to_axis_angle(rotation_6d_to_matrix(poses_6d))   #(T,21,3)
    # print("root_rot, poses, trans", root_rot.shape, poses.shape, trans.shape)   # torch.Size([46, 6]) torch.Size([46, 21, 3]) torch.Size([46, 3])
    poses = torch.cat([root_rot.unsqueeze(2), poses], dim=2).reshape((motion.shape[0],-1,22*3))       #(T,22,3)

    return trans, poses