# Deep Compositional Phase Diffusion for Long Motion Sequence Generation (NeurIPS 2025 Oral)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9.13](https://img.shields.io/badge/python-3.9.13-blue.svg)](https://www.python.org/downloads/release/python-3913/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-<2510.14427>-<COLOR>.svg)](https://arxiv.org/abs/2510.14427)
[![Supplementary](https://img.shields.io/badge/Supplementary%20Material-grey?style=flat&logo=Files&logoColor=white)](https://drive.google.com/file/d/1SNcRA188Or7ZRD_99S-7Dp_Yqn_vQXAO/view?usp=sharing)

<!--Supp: https://drive.google.com/file/d/1SNcRA188Or7ZRD_99S-7Dp_Yqn_vQXAO/view?usp=sharing -->

![teaser](https://asdryau.github.io/TransPhase/static/images/Figure1.jpg)

## 🎯 Abstract
<b>TL;DR</b>
> The proposed Compositional Phase Diffusion framework consistently generates semantically aligned multi-clip motion with smooth transitions by using latent-phase diffusion modules (SPDM and TPDM) to preserve phase continuity and enable inbetweening.

<details><summary><b>CLICK for full abstract</b></summary>

> Recent research on motion generation has shown significant progress in generating semantically aligned motion with singular semantics. However, when employing these models to create composite sequences containing multiple semantically generated motion clips, they often struggle to preserve the continuity of motion dynamics at the transition boundaries between clips, resulting in awkward transitions and abrupt artifacts. To address these challenges, we present Compositional Phase Diffusion, which leverages the Semantic Phase Diffusion Module (SPDM) and Transitional Phase Diffusion Module (TPDM) to progressively incorporate semantic guidance and phase details from adjacent motion clips into the diffusion process. Specifically, SPDM and TPDM operate within the latent motion frequency domain established by the pre-trained Action-Centric Motion Phase Autoencoder (ACT-PAE). This allows them to learn semantically important and transition-aware phase information from variable-length motion clips during training. Experimental results demonstrate the competitive performance of our proposed framework in generating compositional motion sequences that align semantically with the input conditions, while preserving phase transitional continuity between preceding and succeeding motion clips. Additionally, motion inbetweening task is made possible by keeping the phase parameter of the input motion sequences fixed throughout the diffusion process, showcasing the potential for extending the proposed framework to accommodate various application scenarios.
</details>

## 📚 Citation

If you find this work helpful in your research, please consider leaving a star ⭐️ and citing:

```bibtex
@inproceedings{au2025transphase,
  title={Deep Compositional Phase Diffusion for Long Motion Sequence Generation},
  author={Au, Ho Yin and Chen, Jie and Jiang, Junkun and Xiang, Jingyu},
  year={2025},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```

Please checkout our follow-up works if interested:

[SOSControl](https://github.com/asdryau/SOSControl/) - saliency-aware and precise control of body part orientation and motion timing in text-to-motion generation.

## 📋 TODO

- ✅ Released model and dataloader code
- ✅ Released model checkpoints and demo script
- ✅ Released processed data along with training and testing instructions
- ✅ Released code for generating evaluation motion samples
- 🔄 Provide detailed instructions and setup for running data processing and evaluation scripts in the external repository


## 🔮 Environment Setup

### Environment Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/asdryau/TransPhase.git
   cd TransPhase
   ```

2. **Create a conda environment**

   ```bash
   conda create -n transphase python=3.9.13
   conda activate transphase
   ```

3. **Install dependencies**

   ```bash
   conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
   pip install -r requirements.txt
   ```

<!-- ## 📊 Data and Model Preparation -->

### Dataset and Pretrained Model

1. **Download**

   - Download `model_weights.zip` and `processed_data.zip` from [HERE](https://drive.google.com/drive/folders/17SFCS2li0zFHhBDDItKDyjd4ldYyuCjF?usp=drive_link)

2. **Repository Setup**

   - Extract both ZIP files and copy the contents into the `TransPhase/` directory of the current repository.

3. **Final File Structure**

   ```bash
   TransPhase
   ├── data
   │   ├──  label_clip_emb_BABELteach.npz
   │   ├──  meta_motion_CLIP_BABELteach_rel_train.json
   │   └──  motion_CLIP_BABELteach_rel_train.pkl
   ├── evaluation
   │   ├──  evaluation_data.csv
   │   └──  evaluation_data.pkl
   ├── model
   │   ├──  PAE/lightning_logs/version_0/checkpoints/last.ckpt
   │   ├──  SPDM/lightning_logs/version_0/checkpoints/last.ckpt
   │   ├──  TPDM/lightning_logs/version_0/checkpoints/last.ckpt
   │   ├──  inv_rand_proj_15.npy
   │   └──  rand_proj_15.npy
   └── utils
       └──  SMPL_FEMALE.pkl
   ```


## 🚀 Motion Synthesis
The input text and duration specifications can be modified directly within the demo script.
1. **Long-term Motion Generation**
```bash
python demo_t2m_long.py
```

2. **Motion Inbetweening**
```bash
python demo_mib.py
```

## 🖥️ Visualization
We use the SMPL-X Blender add-on to visualize the generated `.npz` file.

Please register at [(https://smpl-x.is.tue.mpg.de)](https://smpl-x.is.tue.mpg.de), download the SMPL-X for Blender add-on, and follow the provided installation instructions.

Once installed, select **Animation -> Add Animation** within the SMPL-X sidebar tool, and navigate to the generated `.npz` file for visualization.

## 🔧 Training

### 1. Train ACT-PAE

```bash
python -m model.PAE.train
```

### 2. Train SPDM and TPDM

```bash
python -m model.SPDM.train
python -m model.TPDM.train
```

**Note:** For details on processing the BABEL-TEACH dataset, please refer to the [PriorMDM data processing script](https://github.com/priorMDM/priorMDM/blob/main/data_loaders/amass/babel.py) and the code snippets in `misc/babel.py` and `model/datamodule_babelteach_rel.py` within this repository for more information.

## 📈 Evaluation
To generate the evaluation output for our model, execute the following commands:
```bash
python -m evaluation.test_mib
python -m evaluation.test_t2m_pair
python -m evaluation.test_t2m_long
```

To run the evaluation for the motion inbetweening task, execute the following commands:
```bash
python -m evaluation.qe_mib
```

**Note:** For details on evaluating on the BABEL-TEACH dataset, please refer to the [PriorMDM evaluation script](https://github.com/priorMDM/priorMDM/blob/main/eval/eval_babel.py#L324) and [PriorMDM evaluation dataloader](https://github.com/priorMDM/priorMDM/blob/main/data_loaders/humanml/motion_loaders/model_motion_loaders.py#L76) for more information.


## 🙏 Acknowledgments

- **[SMPL/SMPL-X](https://smpl.is.tue.mpg.de/)**: For human body modeling (SMPL_Female.pkl) 
- **[PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/v0.3.0/pytorch3d/transforms/rotation_conversions.py)**: For rotation conversion utilities
- **[BABEL-TEACH Dataset](https://github.com/atnikos/teach)**: For motion-text paired data
- **[PriorMDM](https://github.com/priorMDM/priorMDM)**: For data processing and text-motion evaluation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
