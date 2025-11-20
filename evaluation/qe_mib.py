import numpy as np
import torch
import pickle
from utils.rotation_conversion import *
from utils.algo import *
from pathlib import Path
import csv

blend_frame = 60
blend_frame = int(blend_frame/2)*2      # blend_frame must be even

all_paths = [
    f"evaluation/generated_repo/interp_baseline-60",\
    # f"evaluation/generated_repo/model_PAE_mixscaleT_3PE_15_mse/eval_mib-120_txt-0_phasemix-1-1blend-original_DTW-1_output",\

    # f"ablation_CMIB6D/generated_repo/eval_mib-{blend_frame}_txt-0_output",\
    # f"ablation_RMIB6D/generated_repo/eval_mib-{blend_frame}_txt-0_output",\
    # f"MDM_nulltrans/generated_repo/eval_mib-{blend_frame}_txt-0_output",\

    # f"evaluation/generated_repo/model_PAE_mixscaleT_3PE_15_mse/eval_mib-{blend_frame}_txt-0_piecewise-1-1_output/blend-original_DTW-1",\
    # f"evaluation/generated_repo/model_PAE_mixscaleT_3PE_15_mse/eval_mib-{blend_frame}_txt-0_piecewise-1-1_output/blend-original_DTW-0",\

    # r"Z:\hyau\AAAI_TransPhase\visualization\comparisons\mib\CMIB\eval_mib-120_txt-0_output",\
    # r"Z:\hyau\AAAI_TransPhase\visualization\comparisons\mib\CMIB2\eval_mib-120_txt-0_output",\

    # r"Z:\hyau\AAAI_TransPhase\visualization\comparisons\mib\RSMT\output_p10_blend_frame_npz".replace("blend_frame", str(blend_frame)),\
    # r"Z:\hyau\AAAI_TransPhase\visualization\comparisons\mib\RSMT\output_p512_60_npz",\
    # r"Z:\hyau\AAAI_TransPhase\visualization\comparisons\mib\RSMT\output_p512_blend_frame_npz".replace("blend_frame", str(blend_frame)),\
    # r"Z:\hyau\AAAI_TransPhase\visualization\comparisons\mib\priorMDM\eval_mib_blend_frame_output".replace("blend_frame", str(blend_frame)),\
    
    # r"Z:\hyau\AAAI_TransPhase\visualization\comparisons\mib\interp_old\mib120",\
    # r"Z:\hyau\AAAI_TransPhase\visualization\comparisons\mib\interp_old\mib120",\
    # r"Z:\hyau\AAAI_TransPhase\visualization\comparisons\mib\TPDM\eval_mib-120_txt-0_phasemix-1-1_output_fwd",\

    # phase mix
    # r"Z:\hyau\AAAI_TransPhase\visualization\ablation\phasemix_mib\eval_mib-30_txt-0_phasemix-0-1_output",\
    # r"Z:\hyau\AAAI_TransPhase\visualization\ablation\phasemix_mib\eval_mib-30_txt-0_phasemix-0to1-1_output",\
    # r"Z:\hyau\AAAI_TransPhase\visualization\ablation\phasemix_mib\eval_mib-30_txt-0_phasemix-0to1cube-1_output",\
    # r"Z:\hyau\AAAI_TransPhase\visualization\ablation\phasemix_mib\eval_mib-30_txt-0_phasemix-1-0_output",\
    # r"Z:\hyau\AAAI_TransPhase\visualization\ablation\phasemix_mib\eval_mib-30_txt-0_phasemix-1-0to1_output",\
    # r"Z:\hyau\AAAI_TransPhase\visualization\ablation\phasemix_mib\eval_mib-30_txt-0_phasemix-1-0to1cube_output",\
    # r"Z:\hyau\AAAI_TransPhase\visualization\ablation\phasemix_mib\eval_mib-30_txt-0_phasemix-1-1_output",\
    # r"Z:\hyau\AAAI_TransPhase\visualization\ablation\phasemix_mib\eval_mib-30_txt-0_phasemix-1-1to0_output",\
    # r"Z:\hyau\AAAI_TransPhase\visualization\ablation\phasemix_mib\eval_mib-30_txt-0_phasemix-1-05_output",\
    # r"Z:\hyau\AAAI_TransPhase\visualization\ablation\phasemix_mib\eval_mib-30_txt-0_phasemix-1-neg1to0cube_output",\
    # r"Z:\hyau\AAAI_TransPhase\visualization\ablation\phasemix_mib\eval_mib-30_txt-0_phasemix-1to0-1_output",\
    # r"Z:\hyau\AAAI_TransPhase\visualization\ablation\phasemix_mib\eval_mib-30_txt-0_phasemix-05-1_output",\
    # r"Z:\hyau\AAAI_TransPhase\visualization\ablation\phasemix_mib\eval_mib-30_txt-0_phasemix-neg1to0cube-1_output",\


    # r"Z:\hyau\AAAI_TransPhase\visualization\ablation\eps_GMD\model_PAE_mixscaleT_3PE_5_mse\eval_mib-30_txt-0_phasemix-1-1_output",\
    # r"Z:\hyau\AAAI_TransPhase\visualization\ablation\eps_GMD\model_PAE_mixscaleT_3PE_10_mse\eval_mib-30_txt-0_phasemix-1-1_output",\
    # r"Z:\hyau\AAAI_TransPhase\visualization\ablation\eps_GMD\model_PAE_mixscaleT_3PE_20_mse\eval_mib-30_txt-0_phasemix-1-1_output",\
    # r"Z:\hyau\AAAI_TransPhase\visualization\ablation\eps_GMD\model_PAE_mixscaleT_3PE_25_mse\eval_mib-30_txt-0_phasemix-1-1_output",\
    
    # r"Z:\hyau\AAAI_TransPhase\visualization\ALL_for_T2M\phase_channel\model_PAE_mixscaleT_3PE_15_mse_16\eval_mib-30_txt-0_phasemix-1-1_output",\
    # # r"Z:\hyau\AAAI_TransPhase\visualization\ALL_for_T2M\phase_channel\model_PAE_mixscaleT_3PE_15_mse_16\eval_mib-60_txt-0_phasemix-1-1_output",\
    # r"Z:\hyau\AAAI_TransPhase\visualization\ALL_for_T2M\phase_channel\model_PAE_mixscaleT_3PE_15_mse_128\eval_mib-30_txt-0_phasemix-1-1_output",\
    # # r"Z:\hyau\AAAI_TransPhase\visualization\ALL_for_T2M\phase_channel\model_PAE_mixscaleT_3PE_15_mse_128\eval_mib-60_txt-0_phasemix-1-1_output",\
    
    ]


def evaluate(motion_path):
    # motion_path = f"evaluation/generated_repo/model_PAE_mixscaleT_3PE_15_mse/DTW_blend_mib/eval_mib-120_txt-0_phasemix-1-1blend-original_DTW-0_output"
    # motion_path = r"Z:\hyau\AAAI_TransPhase\visualization\comparisons\cmib\CMIB\eval_mib-120_txt-1_output"
    motion_path = Path(motion_path)
    print(motion_path.stem)

    ########################################################################################################
    # reference: https://github.com/cr7anand/neural_temporal_models/blob/master/metrics.py
    def compute_npss(euler_gt_sequences, euler_pred_sequences):

        # computing 1) fourier coeffs 2)power of fft 3) normalizing power of fft dim-wise 4) cumsum over freq. 5) EMD 
        gt_fourier_coeffs = np.zeros(euler_gt_sequences.shape)
        pred_fourier_coeffs = np.zeros(euler_pred_sequences.shape)
        
        # power vars
        gt_power = np.zeros((gt_fourier_coeffs.shape))
        pred_power = np.zeros((gt_fourier_coeffs.shape))
        
        # normalizing power vars
        gt_norm_power = np.zeros(gt_fourier_coeffs.shape)
        pred_norm_power = np.zeros(gt_fourier_coeffs.shape)
        
        cdf_gt_power = np.zeros(gt_norm_power.shape)
        cdf_pred_power = np.zeros(pred_norm_power.shape)
        
        emd = np.zeros(cdf_pred_power.shape[0:3:2])
        
        # used to store powers of feature_dims and sequences used for avg later
        seq_feature_power = np.zeros(euler_gt_sequences.shape[0:3:2])
        power_weighted_emd = 0
        
        for s in range(euler_gt_sequences.shape[0]):
            
            for d in range(euler_gt_sequences.shape[2]):
                gt_fourier_coeffs[s,:,d] = np.fft.fft(euler_gt_sequences[s,:,d]) # slice is 1D array
                pred_fourier_coeffs[s,:,d] = np.fft.fft(euler_pred_sequences[s,:,d])

                # computing power of fft per sequence per dim
                gt_power[s,:,d] = np.square(np.absolute(gt_fourier_coeffs[s,:,d]))
                pred_power[s,:,d] = np.square(np.absolute(pred_fourier_coeffs[s,:,d]))
                
                # matching power of gt and pred sequences
                gt_total_power = np.sum(gt_power[s,:,d])
                pred_total_power = np.sum(pred_power[s,:,d])
                #power_diff = gt_total_power - pred_total_power
                
                # adding power diff to zero freq of pred seq
                #pred_power[s,0,d] = pred_power[s,0,d] + power_diff
                
                # computing seq_power and feature_dims power 
                seq_feature_power[s,d] = gt_total_power
                
                # normalizing power per sequence per dim
                if gt_total_power != 0:
                    gt_norm_power[s,:,d] = gt_power[s,:,d] / gt_total_power 
                
                if pred_total_power !=0:
                    pred_norm_power[s,:,d] = pred_power[s,:,d] / pred_total_power
        
                # computing cumsum over freq
                cdf_gt_power[s,:,d] = np.cumsum(gt_norm_power[s,:,d]) # slice is 1D
                cdf_pred_power[s,:,d] = np.cumsum(pred_norm_power[s,:,d])
        
                # computing EMD 
                emd[s,d] = np.linalg.norm((cdf_pred_power[s,:,d] - cdf_gt_power[s,:,d]), ord=1)

        # computing weighted emd (by sequence and feature powers)	
        power_weighted_emd = np.average(emd, weights=seq_feature_power) 

        return power_weighted_emd


    # reference: https://github.com/jihoonerd/Conditional-Motion-In-Betweening/blob/40f42c6d2d0e081e2162569180c5e2ad42ce659e/cmib/model/preprocess.py
    def slerp(x, y, a):
        """
        Perfroms spherical linear interpolation (SLERP) between x and y, with proportion a

        :param x: quaternion tensor
        :param y: quaternion tensor
        :param a: indicator (between 0 and 1) of completion of the interpolation.
        :return: tensor of interpolation results
        """
        device = x.device
        len = torch.sum(x * y, dim=-1)

        neg = len < 0.0
        len[neg] = -len[neg]
        y[neg] = -y[neg]

        a = torch.zeros_like(x[..., 0]) + a
        amount0 = torch.zeros(a.shape, device=device)
        amount1 = torch.zeros(a.shape, device=device)

        linear = (1.0 - len) < 0.01
        omegas = torch.arccos(len[~linear])
        sinoms = torch.sin(omegas)

        amount0[linear] = 1.0 - a[linear]
        amount0[~linear] = torch.sin((1.0 - a[~linear]) * omegas) / sinoms

        amount1[linear] = a[linear]
        amount1[~linear] = torch.sin(a[~linear] * omegas) / sinoms
        # res = amount0[..., np.newaxis] * x + amount1[..., np.newaxis] * y
        res = amount0.unsqueeze(-1) * x + amount1.unsqueeze(-1) * y

        return res

    ########################################################################################################


    motion_path_avail = motion_path.rglob("*.npz")
    motion_path_avail = [f.stem for f in motion_path_avail]

    # generated_motions = motion_path.rglob("*.npz")
    # generated_motions = [m for m in generated_motions]

    with open('evaluation/evaluation_data.pkl', 'rb') as file:  
        save_pkl = pickle.load(file)

    keys_all, poses_all, trans_all, time_all, random_text_all, orig_text_pair_all =\
        save_pkl["keys"], save_pkl["poses"], save_pkl["trans"], save_pkl["time"], save_pkl["random_text"], save_pkl["orig_text_pair"]

    processed_data = list(zip(*[keys_all, poses_all, trans_all, time_all, random_text_all, orig_text_pair_all]))


    generated_rotvel_all = []
    poses_rotvel_all = []
    generated_rotacc_all = []
    poses_rotacc_all = []
    generated_rotjerk_all = []
    poses_rotjerk_all = []
    l2q_all = []
    l26d_all = []
    l2p_all = []
    l2p_noroot_all = []
    l2p_norootrot_all = []
    l2v_all = []
    l2v_noroot_all = []
    l2v_norootrot_all = []
    npss_all = []
    npss6d_all = []
    npsspos_all = []
    l2q_lerp_all = []
    l2p_lerp_all = []

    with torch.no_grad():
        ### eval script
        for k in processed_data:
            # text is ignored
            keys, poses, trans, [pl, pe, se], random_text, _ = k

            ### define motion segment
                # poses[pl:pe] is preceding motion
                # poses[pe:se] is succeeding motion (note that pe==sl)
            sl = pe

            se = se - 1
            
            ### check if there exists sufficient space for mib evaluation
                # after masking, each motion should have at least 30 frames
            blend_frame_half = int(blend_frame/2)
            if (pe-pl) - blend_frame_half < 30 or (se-sl) - blend_frame_half < 30:
                continue
            blend_start = pe-blend_frame_half
            blend_end = sl+blend_frame_half
            
            if not (keys in motion_path_avail):
                continue

            #   !!!!! read generated motion
            generated = np.load(motion_path / f"{keys}.npz")
            generated_poses = torch.from_numpy(generated["poses"]).float()
            generated_trans = torch.from_numpy(generated["trans"]).float()  #(T,3)
            # print("generated", generated.shape)
            
            poses = poses[pl:se,:22,:]  #(T,21,3)
            generated_poses = generated_poses[pl:se,:22*3].reshape((-1,22,3))      #(T,21,3)  

            ###   process GT motion
                # check only 21 joints, check only blend_start-blend_end to evaluate the blended segment
            poses = matrix_to_axis_angle(rotation_6d_to_matrix(poses))
            poses_end1 = poses[blend_start-1,:22,:]  #(22,3)
            poses_end1 = axis_angle_to_quaternion(poses_end1).unsqueeze(0)
            poses_start2 = poses[blend_end,:22,:]  #(22,3)
            poses_start2 = axis_angle_to_quaternion(poses_start2).unsqueeze(0)
            
            poses = axis_angle_to_quaternion(poses)
            generated_poses = axis_angle_to_quaternion(generated_poses)                 #(T,21,4)

            trans = trans[pl:se]
            trans = trans - trans[0:1]
            generated_trans = generated_trans[pl:se]
            generated_trans = generated_trans - generated_trans[0:1]

            ###   generate interp motion
            T = blend_end - blend_start
            # lerp trans
            alpha = torch.linspace(0,1,T+2).unsqueeze(-1)  #(T,1)
            lerp_trans = trans[blend_start-1:blend_start] * (1-alpha) + trans[blend_end:blend_end+1] * alpha
            lerp_trans = lerp_trans[1:-1]
            lerp_trans = torch.cat([trans[:blend_start], lerp_trans, trans[blend_end:]], dim=0)
            # slerp poses
            slerp_poses = []
            for t in range(T+2):
                # T+2 as we interp boundary pose
                slerp_poses.append(slerp(poses_end1, poses_start2, t/(T-1+2)))
            slerp_poses = torch.cat(slerp_poses, dim=0)[1:-1]   # remove the boundary pose as it is not included in inb_region
            slerp_poses = torch.cat([poses[:blend_start], slerp_poses, poses[blend_end:]], dim=0)            


            ### just slice again according to spec
            poses = poses[blend_start:blend_end]
            generated_poses = generated_poses[blend_start:blend_end]
            slerp_poses = slerp_poses[blend_start:blend_end]
            
            trans = trans[blend_start:blend_end]
            trans = trans - trans[0:1]
            generated_trans = generated_trans[blend_start:blend_end]
            generated_trans = generated_trans - generated_trans[0:1]
            lerp_trans = lerp_trans[blend_start:blend_end]
            lerp_trans = lerp_trans - lerp_trans[0:1]


            ###
            #   metrics 1: average joint update
            ###
                # calc root difference per frame
            generated_rotvel = quaternion_multiply(generated_poses[1:], quaternion_invert(generated_poses[:-1]))
            generated_rotacc = quaternion_multiply(generated_rotvel[1:], quaternion_invert(generated_rotvel[:-1]))
            generated_rotjerk = quaternion_multiply(generated_rotacc[1:], quaternion_invert(generated_rotacc[:-1]))
            generated_rotvel = quaternion_to_axis_angle(generated_rotvel)
            generated_rotacc = quaternion_to_axis_angle(generated_rotacc)
            generated_rotjerk = quaternion_to_axis_angle(generated_rotjerk)
                # the (axis-)angle of diff_rotation is the angular difference
            # generated_rotvel = torch.norm(generated_rotvel, dim=2)      #(T,21)
            generated_rotvel = torch.mean(torch.square(generated_rotvel)).unsqueeze(0)                   #(1,)
            # generated_rotacc = torch.norm(generated_rotacc, dim=2)      #(T,21)
            generated_rotacc = torch.mean(torch.square(generated_rotacc)).unsqueeze(0)                   #(1,)
            generated_rotjerk = torch.mean(torch.square(generated_rotjerk)).unsqueeze(0)                   #(1,)

                # calc root difference per frame
            poses_rotvel = quaternion_multiply(poses[1:], quaternion_invert(poses[:-1]))
            poses_rotacc = quaternion_multiply(poses_rotvel[1:], quaternion_invert(poses_rotvel[:-1]))
            poses_rotjerk = quaternion_multiply(poses_rotacc[1:], quaternion_invert(poses_rotacc[:-1]))
            poses_rotvel = quaternion_to_axis_angle(poses_rotvel)
            poses_rotacc = quaternion_to_axis_angle(poses_rotacc)
            poses_rotjerk = quaternion_to_axis_angle(poses_rotjerk)
                # the (axis-)angle of diff_rotation is the angular difference
            # poses_rotvel = torch.norm(poses_rotvel, dim=2)        #(T,21)
            poses_rotvel = torch.mean(torch.square(poses_rotvel)).unsqueeze(0)                 #(1,)
            # poses_rotacc = torch.norm(poses_rotacc, dim=2)        #(T,21)
            poses_rotacc = torch.mean(torch.square(poses_rotacc)).unsqueeze(0)                 #(1,)
            poses_rotjerk = torch.mean(torch.square(poses_rotjerk)).unsqueeze(0)                 #(1,)

            generated_rotvel_all.append(generated_rotvel)
            poses_rotvel_all.append(poses_rotvel)

            generated_rotacc_all.append(generated_rotacc)
            poses_rotacc_all.append(poses_rotacc)

            generated_rotjerk_all.append(generated_rotjerk)
            poses_rotjerk_all.append(poses_rotjerk)

            # ###
            # #   metrics 2: L2Q
            # ###
            # l2q = torch.norm(generated_poses - poses, dim=2)       #(T,21)
            # l2q = torch.mean(l2q).unsqueeze(0)                 #(1,)
            l2q = torch.sqrt(torch.mean(torch.square(generated_poses - poses))).unsqueeze(0)                 #(1,)
            # l2q = torch.mean(torch.square(generated_poses[:,1:] - poses[:,1:])).unsqueeze(0)                 #(1,)
            l2q_all.append(l2q)

            # l26d = torch.norm(matrix_to_rotation_6d(quaternion_to_matrix(generated_poses)) - matrix_to_rotation_6d(quaternion_to_matrix(poses)), dim=2)       #(T,21)
            # l26d = torch.mean(l26d).unsqueeze(0)                 #(1,)
            l26d = torch.sqrt(torch.mean(torch.square(matrix_to_rotation_6d(quaternion_to_matrix(generated_poses)) - matrix_to_rotation_6d(quaternion_to_matrix(poses))))).unsqueeze(0)       #(T,21)
            # l26d = torch.mean(torch.square(matrix_to_rotation_6d(quaternion_to_matrix(generated_poses[:,1:])) - matrix_to_rotation_6d(quaternion_to_matrix(poses[:,1:])))).unsqueeze(0)       #(T,21)
            l26d_all.append(l26d)
            
            # ###
            # #   metrics 3: L2P
            # ###
            global_positions = forward_kinematics(trans, quaternion_to_matrix(poses)) #(T,J,3)
            generated_positions = forward_kinematics(generated_trans, quaternion_to_matrix(generated_poses)) #(T,J,3)
            l2p = generated_positions - global_positions       #(T,21)
            l2v = l2p[1:] - l2p[:-1]                            #(T,21)
            l2p = torch.sqrt(torch.mean(torch.square(l2p)).unsqueeze(0))                 #(1,)
            l2v = torch.sqrt(torch.mean(torch.square(l2v)).unsqueeze(0))                 #(1,)
            l2p_all.append(l2p)
            l2v_all.append(l2v)

            global_positions_noroot = forward_kinematics(torch.zeros_like(trans), quaternion_to_matrix(poses)) #(T,J,3)
            generated_positions_noroot = forward_kinematics(torch.zeros_like(generated_trans), quaternion_to_matrix(generated_poses)) #(T,J,3)
            l2p_noroot = generated_positions_noroot - global_positions_noroot       #(T,21)
            l2v_noroot = l2p_noroot[1:] - l2p_noroot[:-1] 
            l2p_noroot = torch.sqrt(torch.mean(torch.square(l2p_noroot)).unsqueeze(0))                 #(1,)
            l2v_noroot = torch.sqrt(torch.mean(torch.square(l2v_noroot)).unsqueeze(0))                 #(1,)
            l2p_noroot_all.append(l2p_noroot)
            l2v_noroot_all.append(l2v_noroot)

            poses_normrootrot = quaternion_to_matrix(poses)
            poses_normrootrot[:,0] = torch.eye(3).type_as(poses)
            generated_poses_normrootrot = quaternion_to_matrix(generated_poses)
            generated_poses_normrootrot[:,0] = torch.eye(3).type_as(generated_poses)
            global_positions_norootrot = forward_kinematics(torch.zeros_like(trans), poses_normrootrot) #(T,J,3)
            generated_positions_norootrot = forward_kinematics(torch.zeros_like(generated_trans),generated_poses_normrootrot) #(T,J,3)
            l2p_norootrot = generated_positions_norootrot - global_positions_norootrot       #(T,21)
            l2v_norootrot = l2p_norootrot[1:] - l2p_norootrot[:-1] 
            l2p_norootrot = torch.sqrt(torch.mean(torch.square(l2p_norootrot))).unsqueeze(0)                 #(1,)
            l2v_norootrot = torch.sqrt(torch.mean(torch.square(l2v_norootrot))).unsqueeze(0)                 #(1,)
            l2p_norootrot_all.append(l2p_norootrot)
            l2v_norootrot_all.append(l2v_norootrot)

            # ###
            # #   metrics 4: NPSS
            # ###
            poses_euler = matrix_to_euler_angles(quaternion_to_matrix(poses), "XYZ").reshape((-1,22*3)).unsqueeze(0)       #(1,T,21*3)
            generated_euler = matrix_to_euler_angles(quaternion_to_matrix(generated_poses), "XYZ").reshape((-1,22*3)).unsqueeze(0)  
            npss = compute_npss(poses_euler.numpy(), generated_euler.numpy())
            npss_all.append(npss)

            poses_6d = matrix_to_rotation_6d(quaternion_to_matrix(poses)).reshape((-1,22*6)).unsqueeze(0)       #(1,T,21*3)
            generated_6d = matrix_to_rotation_6d(quaternion_to_matrix(generated_poses)).reshape((-1,22*6)).unsqueeze(0)  
            npss6d = compute_npss(poses_6d.numpy(), generated_6d.numpy())
            npss6d_all.append(npss6d)

            poses_pos = global_positions_noroot.permute(1,0,2)       #.reshape((-1,22*6)).unsqueeze(0)       #(1,T,21*3)
            generated_pos = generated_positions_noroot.permute(1,0,2)      #.reshape((-1,22*6)).unsqueeze(0)  
            npsspos = compute_npss(poses_pos.numpy(), generated_pos.numpy())
            npsspos_all.append(npsspos)
            
            ###
            #   metrics 5: Abruptness
            ###
            # l2q_lerp = torch.norm(generated_poses - slerp_poses, dim=2)       #(T,21)
            # l2q_lerp = torch.mean(l2q_lerp).unsqueeze(0)                 #(1,)
            l2q_lerp = torch.sqrt(torch.mean(torch.square(generated_poses - slerp_poses))).unsqueeze(0)                 #(1,)
            l2q_lerp_all.append(l2q_lerp)

            lerp_positions = forward_kinematics(lerp_trans, quaternion_to_matrix(slerp_poses)) #(T,J,3)
            # l2p_lerp = torch.norm(generated_positions - lerp_positions, dim=2)       #(T,21)
            # l2p_lerp = torch.mean(l2p_lerp).unsqueeze(0)                 #(1,)
            l2p_lerp = torch.sqrt(torch.mean(torch.square(generated_positions - lerp_positions))).unsqueeze(0)                 #(1,)
            l2p_lerp_all.append(l2p_lerp)

        generated_rotvel_all = torch.cat(generated_rotvel_all, dim=0)
        poses_rotvel_all = torch.cat(poses_rotvel_all, dim=0)
        generated_rotacc_all = torch.cat(generated_rotacc_all, dim=0)
        poses_rotacc_all = torch.cat(poses_rotacc_all, dim=0)
        generated_rotjerk_all = torch.cat(generated_rotjerk_all, dim=0)
        poses_rotjerk_all = torch.cat(poses_rotjerk_all, dim=0)
        l2p_all = torch.cat(l2p_all, dim=0)
        l2p_noroot_all = torch.cat(l2p_noroot_all, dim=0)
        l2p_norootrot_all = torch.cat(l2p_norootrot_all, dim=0)
        l2v_all = torch.cat(l2v_all, dim=0)
        l2v_noroot_all = torch.cat(l2v_noroot_all, dim=0)
        l2v_norootrot_all = torch.cat(l2v_norootrot_all, dim=0)
        l2q_all = torch.cat(l2q_all, dim=0)
        l26d_all = torch.cat(l26d_all, dim=0)
        l2p_lerp_all = torch.cat(l2p_lerp_all, dim=0)
        l2q_lerp_all = torch.cat(l2q_lerp_all, dim=0)

        metrics = {}
        metrics["generated average velocity"] = generated_rotvel_all.mean() * 30
        metrics["GT average velocity"] = poses_rotvel_all.mean() * 30
        metrics["generated average accel"] = generated_rotacc_all.mean() * 30
        metrics["GT average accel"] = poses_rotacc_all.mean() * 30
        metrics["generated average jerk"] = generated_rotjerk_all.mean() * 30
        metrics["GT average jerk"] = poses_rotjerk_all.mean() * 30
        metrics["L2P"] = l2p_all.mean()
        metrics["L2P_noroot"] = l2p_noroot_all.mean()
        metrics["L2P_norootrot"] = l2p_norootrot_all.mean()
        metrics["L2V"] = l2v_all.mean()
        metrics["L2V_noroot"] = l2v_noroot_all.mean()
        metrics["L2V_norootrot"] = l2v_norootrot_all.mean()
        metrics["L2Q"] = l2q_all.mean()
        metrics["L26D"] = l26d_all.mean()
        metrics["NPSS"] = np.mean(npss_all)
        metrics["NPSS_6D"] = np.mean(npss6d_all)
        metrics["NPSS_POS"] = np.mean(npsspos_all)
        metrics["L2P_lerp"] = l2p_lerp_all.mean()
        metrics["L2Q_lerp"] = l2q_lerp_all.mean()

        
        savefilename = str(motion_path) + "_mod.csv"
        # with open(savefilename, 'w') as savefile:
        #     for k in metrics:
        #         savefile.write(k + '\t:\t' + "{:10.4f}".format(float(metrics[k])) + '\n')

        with open(savefilename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(list(metrics.keys()))
            writer.writerow([float(v) for v in list(metrics.values())])



for p in all_paths:
    evaluate(p)