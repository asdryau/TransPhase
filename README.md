# TransPhase: Transition Phase Motion Generation with Diffusion Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9.13](https://img.shields.io/badge/python-3.9.13-blue.svg)](https://www.python.org/downloads/release/python-3913/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

TransPhase is a deep learning framework for generating smooth transition phases in human motion sequences using diffusion models and Parametric AutoEncoders (PAE). The system focuses on generating coherent motion transitions between different motion segments, particularly useful for character animation, robotics, and motion synthesis applications.

<!-- ## 🎯 Overview

This project implements a multi-stage approach to human motion generation:

1. **PAE (Parametric AutoEncoder)**: Encodes motion sequences into a compact latent representation using sinusoidal parameterization
2. **SPDM (Sequential Phase Diffusion Model)**: Generates motion phases in sequence
3. **TPDM (Transition Phase Diffusion Model)**: Generates smooth transitions between motion segments

The framework uses the BABEL-Teach dataset and leverages CLIP embeddings for text-conditioned motion generation.

## 🏗️ Architecture

### Key Components

- **MotionPAE**: A transformer-based autoencoder that learns compact motion representations using parametric sine wave functions
- **DiffPhase**: Sequential diffusion model for generating left and right motion phases
- **TranPhase**: Cross-attention based diffusion model for generating transition phases between motion segments

### Technical Features

- **3-way Positional Encoding**: Custom positional encoding for handling variable-length motion sequences
- **GMD Projection**: Guided Motion Diffusion projection for emphasis on important motion features
- **CLIP Integration**: Text-conditional motion generation using CLIP embeddings
- **Classifier-free Guidance**: Improved generation quality through conditional training -->

## 📋 Requirements

### Dependencies

```txt
pytorch-lightning==1.9.0
smplx==0.1.28
scipy==1.10.1
numpy==1.23.1
chumpy==0.70
git+https://github.com/openai/CLIP.git
tensorboard==2.10.0
diffusers==0.34.0
```

### Hardware Requirements

- CUDA-capable GPU (recommended: RTX 3080 or better)
- At least 16GB RAM
- 50GB+ free disk space for datasets and checkpoints

## 🚀 Installation

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
   (when conda takes too long, try pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116)
   pip install -r requirements.txt
   ```

<!-- 4. **Download SMPL models**
   - Download SMPL models from [SMPL website](https://smpl.is.tue.mpg.de/)
   - Place `SMPL_FEMALE.pkl` in the `utils/` directory -->

## 📊 Dataset and Other Dependencies Setup

### Dataset and SMPL Model

1. **Download**

   - Download `model_weights.zip` and `processed_data.zip` from [HERE](https://drive.google.com/drive/folders/16kPBUzQu-xsPHI7Jfkn2DWwsB9F8bnKH?usp=drive_link)

2. **Data Structure**

   ```txt
   model_weights/
   └── TransPhase/
       ├── model/*
       └── utils/*
   ```

    ```txt
   processed_data/
   └── TransPhase/
       ├── data/*
       └── evaluation/*
   ```

3. **Usage**

   - Copy the files under `TransPhase/` from both ZIPs and paste them to your project root path.
<!-- ### Data Format

The dataset contains motion sequences with:

- **Motion data**: SMPL pose parameters and translations  
- **Text annotations**: Natural language descriptions of actions
- **Temporal segments**: Left motion, transition, and right motion phases -->

## 🎮 Run

### Training

#### 1. Train the Parametric AutoEncoder (PAE)

```bash
cd TransPhase
python -m model.PAE.train
```

#### 2. Train Sequential Phase Diffusion Model (SPDM)

```bash
cd TransPhase 
python -m model.SPDM.train
```

#### 3. Train Transition Phase Diffusion Model (TPDM)

```bash
cd TransPhase
python -m model.TPDM.train
```

<!-- ### Configuration

Key training parameters can be modified in the training scripts:

```python
# Training configuration
batch_size = 2                    # Batch size for training
latent_dim = 512                  # Latent dimension for motion encoding
max_epochs = 300000               # Maximum training epochs
num_train_timesteps = 1000        # Diffusion timesteps
guidance_scale = 7.5              # Classifier-free guidance scale
``` -->

### Inference

```python
# Load trained models
from model.PAE.model import MotionPAE
from model.TPDM.diffusion import TranPhase

# Initialize models
pae = MotionPAE.load_from_checkpoint("path/to/pae/checkpoint.ckpt")
transphase = TranPhase.load_from_checkpoint("path/to/transphase/checkpoint.ckpt")

# Generate motion transitions
# (Implementation details depend on specific use case)
```

## 📁 Project Structure

```txt
TransPhase/
├── model/
│   ├── datamodule_babelteach_rel.py    # Data loading and preprocessing
│   ├── PAE/
│   │   ├── model.py                     # Parametric AutoEncoder
│   │   └── train.py                     # PAE training script
│   ├── SPDM/
│   │   ├── diffusion.py                 # Sequential Phase Diffusion Model  
│   │   └── train.py                     # SPDM training script
│   └── TPDM/
│       ├── diffusion.py                 # Transition Phase Diffusion Model
│       └── train.py                     # TPDM training script
├── utils/
│   ├── algo.py                          # Motion processing algorithms
│   └── rotation_conversion.py           # Rotation utilities (from PyTorch3D)
├── LICENSE                              # MIT License
└── README.md                            # This file
```

<!-- ## 🔧 Key Features

### Motion Representation

- **SMPL-based**: Uses SMPL body model for realistic human motion
- **6D Rotation**: Robust rotation representation using 6D rotation matrices  
- **Velocity Features**: Incorporates motion velocities and foot contacts
- **Root-relative**: Motion represented relative to pelvis root joint

### Diffusion Models

- **DDIM/DDPM Schedulers**: Flexible noise scheduling for training and inference
- **Multi-scale Architecture**: Transformer-based denoisers with attention mechanisms
- **Conditional Generation**: Text-guided motion generation using CLIP embeddings

### Training Techniques

- **Progressive Training**: Multi-stage training from autoencoder to diffusion models
- **Mixed Precision**: FP16 training for efficiency
- **Multi-GPU Support**: Distributed training across multiple GPUs

## 📈 Performance

The model is designed to generate high-quality motion transitions with:

- **Temporal Coherence**: Smooth transitions between motion phases
- **Semantic Consistency**: Motion matches text descriptions
- **Physical Plausibility**: Realistic human motion dynamics -->

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SMPL/SMPL-X**: For human body modeling
- **CLIP**: For text-motion alignment  
- **PyTorch3D**: For rotation conversion utilities
- **BABEL Dataset**: For motion-text paired data
- **Diffusers**: For diffusion model implementations
- **PyTorch Lightning**: For training framework

## 📚 References

If you use this work in your research, please consider citing:

```bibtex
@misc{transphase2025,
  title={TransPhase: Transition Phase Motion Generation with Diffusion Models},
  author={Ryan Au},
  year={2025},
  url={https://github.com/asdryau/TransPhase}
}
```

## 🐛 Issues & Support

If you encounter any issues or have questions:

1. Check existing [Issues](https://github.com/asdryau/TransPhase/issues)
2. Create a new issue with detailed description
3. Include system information and error logs

<!-- ## 🔮 Future Work

- [ ] Real-time motion generation optimization
- [ ] Multi-character interaction modeling  
- [ ] Extended motion style control
- [ ] Integration with physics simulation
- [ ] Mobile/edge device deployment -->
