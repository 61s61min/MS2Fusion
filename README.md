# Multispectral State-Space Feature Fusion: Bridging Shared and Cross-Parametric Interactions for Object Detection

## Abstract   
Modern multispectral feature fusion for object detection faces two critical limitations: (1) Excessive preference for local complementary features over cross-modal shared semantics adversely affects generalization performance; and (2) The trade-off between the receptive field size and computational complexity present critical bottlenecks for scalable feature modeling. Addressing these issues, a novel Multispectral State-Space Feature Fusion framework, dubbed MS2Fusion, is proposed based on the state space model (SSM), achieving efficient and effective fusion through a dual-path parametric interaction mechanism. More specifically, the first cross-parameter interaction branch inherits the advantage of cross-attention in mining complementary information with cross-modal hidden state decoding in SSM. The second shared-parameter branch explores cross-modal alignment with joint embedding to obtain cross-modal similar semantic features and structures through parameter sharing in SSM. Finally, these two paths are jointly optimized with SSM for fusing multispectral features in a unified framework, allowing our MS2Fusion to enjoy both functional complementarity and shared semantic space. In our extensive experiments on mainstream benchmarks including FLIR, M3FD and LLVIP, our MS2Fusion significantly outperforms other state-of-the-art multispectral object detection methods, evidencing its superiority. Moreover, MS2Fusion is general and applicable to other multispectral perception tasks. We show that, even without specific design, MS2Fusion achieves state-of-the-art results on RGB-T semantic segmentation and RGBT salient object detection, showing its generality.    
Paper download in [here](https://arxiv.org/abs/2507.14643)

### Overview
<div align="center">
  <img src="overreview.png" width="1200px">
  <div style="color:orange; border-bottom: 10px solid #d9d9d9; display: inline-block; color: #999; padding: 10px;"> Fig 1. Overview of our multispectral object detection framework </div>
</div>


## Acknowledgement  
This is our preliminary code. We will provide a refined, clear, and detailed version after the paper is published. Please contact us if you have any questions.


## Installation
```
git clone https://github.com/61s61min/S3Fusion.git
pip install -r requirements.txt
```
## Install Mamba
Please refer to **[VMamba](https://github.com/MzeroMiko/VMamba.git)**.
## Run

- We use pre-training weights in the training process of RGB-T OD and RGB-T SOD, and the relevant pre-training weights can be found in the **[RGB-T OD](https://pan.baidu.com/s/1qQ4toibx3ikeOEy3WwBnlQ?pwd=ave2)** (ave2), **[RGB-T SOD](https://pan.baidu.com/s/1EhJbit7g_4Q9kcR7I3rNtg?pwd=mz48)**(mz48).  

- You just need to check the path settings in the code to make it run. For more details, please refer to the following **[repository](#thanks)**.

## Dataset
- **FLIR**  
 Link:  https://pan.baidu.com/s/19TbR2PDTE3a-b6aG-wzYdA?pwd=q4gb (q4gb)    
- **LLVIP**  
 Link:  https://pan.baidu.com/s/1AwJIRVtdtRUB1cibz5JilQ?pwd=nfnr (nfnr)  
- **M3FD**  
 Link:  https://github.com/dlut-dimt/TarDAL.git    
- **VEDAI**  
 Link:  https://downloads.greyc.fr/vedai/    
- **MFNet**  
 Link:  https://pan.baidu.com/s/1QkTA6FTTIMhlwbZfPEjqcA?pwd=1odx (1odx)    
- **SemanticRT**  
 Link: https://pan.baidu.com/s/13b7FDIO4FxN98xGAliYEIA?pwd=6rl5 (6rl5)    
- **VT821, VT1000, VT5000**  
 Link: https://pan.baidu.com/s/1Ousl6r2VLpzMowPFOyNRNA?pwd=h0cg (h0cg)     
## Weight
- **FLIR**  
 Link: https://pan.baidu.com/s/1Iie8qUHbN5gnAPCy7uWJhA?pwd=53cw (53cw)   
- **LLVIP**  
Link: https://pan.baidu.com/s/11I2QY6z0kgTumJw6MdJ4Sw?pwd=rtpl (rtpl)   
- **M3FD**  
Link: https://pan.baidu.com/s/1ba4r5cgOnoatf-vcdwClgw?pwd=qpwq (qpwq)
- **VEDAI**  
 Link:  https://pan.baidu.com/s/1Ysq7tyU-whBAvQvQNtT1Aw?pwd=si47 (si47)
- **MFNet**  
 Link:  https://pan.baidu.com/s/1kLcTuoUXr5AKi0yWg81pfw?pwd=tijf (tijf)
- **SemanticRT**  
 Link:  https://pan.baidu.com/s/14ydinnSOzKnuuJprjlZH5g?pwd=vmmz (vmmz)  
- **VT821, VT1000, VT5000**  
 Link:  https://pan.baidu.com/s/1G-ZAk0T67tDALeJHaX3aNQ?pwd=eqe4 (eqe4)



## SOD Salient Map
Link: https://pan.baidu.com/s/1c9ClfwWBoe3o106_OBkyqQ?pwd=r0a5 (r0a5)

## Thanks
Our RGB-T OD code is based on **[ICAFusion](https://github.com/chanchanchan97/ICAFusion.git)**, RGB-T SS code is based on **[MFNet](https://github.com/haqishen/MFNet-pytorch.git)**, and RGB-T SOD code is based on **[MSEDNET](https://github.com/Zhou-wy/MSEDNet.git)**, SOD was evaluated with codes from **[PySODMetrics](https://github.com/lartpang/PySODMetrics.git)** and their contributions are greatly appreciated!
We will update the code later

## Concat us
If you have any other questions about the code, please email **[Haibo Zhan](mailto:haibozhan@outlook.com)**.

## Cite us
@article{shen2025multispectral,     
  title={Multispectral state-space feature fusion: Bridging shared and cross-parametric interactions for object detection},     
  author={Shen, Jifeng and Zhan, Haibo and Dong, Shaohua and Zuo, Xin and Yang, Wankou and Ling, Haibin},      
  journal={arXiv preprint arXiv:2507.14643},        
  year={2025}   
}