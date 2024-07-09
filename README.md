<div  align=center><img src="./assets/logo.png" width="15%"></div>


## <p align=center>[ECCV 2024] Sparse Beats Dense: Rethinking Supervision in Radar-Camera Depth Completion</p>

<p align=center>Huadong Li<sup>*</sup>, Minhao Jing<sup>*</sup>, Jing Wang, Shichao Dong, Jiajun Liang, Haoqiang Fan, Renhe Ji<sup>‡</sup> </p>

**<p align=center>MEGVII Technology</p>**

  <p align=center><sup>*</sup>Equal contribution  <sup>†</sup>Lead this project <sup>‡</sup>Corresponding author</p>


  <div align="center">
  <br>
  <a href='https://arxiv.org/abs/2312.00844'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
  <!-- <a href='https://megactor.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
  <a href='https://f4c5-58-240-80-18.ngrok-free.app/'><img src='https://img.shields.io/badge/DEMO-RUNNING-<COLOR>.svg'></a>
  <a href='https://openbayes.com/console/public/tutorials/3IphFlojVlO'><img src='https://img.shields.io/badge/CONTAINER-OpenBayes-blue.svg'></a> -->
  <br>
</div>


## Overview

  ![Model](./assets/intro.png)

It is widely believed that sparse supervision is worse than dense supervision in the field of depth completion, but the underlying reasons for this are rarely discussed.
To this end, we revisit the task of radar-camera depth completion and present a new method with **sparse LiDAR** supervision to outperform previous **dense LiDAR** supervision methods in both accuracy and speed.

Specifically, when trained by sparse LiDAR supervision, depth completion models usually output depth maps containing significant stripe-like artifacts.
We find that such a phenomenon is caused by the implicitly learned positional distribution pattern from sparse LiDAR supervision, termed as **LiDAR Distribution Leakage (LDL)** in this paper.
Based on such understanding, we present a novel **Disruption-Compensation** radar-camera depth completion framework to address this issue.
The **Disruption** part aims to deliberately disrupt the learning of LiDAR distribution from sparse supervision, while the **Compensation** part aims to leverage 3D spatial and 2D semantic information to compensate for the information loss of previous disruptions.

<!-- By reducing the LDL, we first present the depth completion model trained by sparse supervision. -->
<!-- 
Extensive experimental results demonstrate that by reducing the impact of LDL, our framework with **sparse supervision** outperforms the state-of-the-art **dense supervision** methods with **11.6%** improvement in Mean Absolute Error (MAE) and **1.6** speedup in Frame Per Second (FPS). -->


## Demo Results

Coming soon.

## Preparation

Coming soon.

## Training

Coming soon.


## Inference

Coming soon.


## BibTeX
```
@misc{li2023sparsebeatsdenserethinking,
      title={Sparse Beats Dense: Rethinking Supervision in Radar-Camera Depth Completion}, 
      author={Huadong Li and Minhao Jing and Jiajun Liang and Haoqiang Fan and Renhe Ji},
      year={2023},
      eprint={2312.00844},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2312.00844}, 
}
```

## Contact
If you have any questions, feel free to open an issue or contact us at lihuadong@megvii.com or jirenhe@megvii.com.
