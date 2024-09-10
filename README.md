# S3Fusion: Shared State Space based Feature Fusion for Multispectral Object Detection
## abstract

### Overview
<div align="center">
  <img src="overrall detail.png" width="600px">
  <div style="color:orange; border-bottom: 10px solid #d9d9d9; display: inline-block; color: #999; padding: 10px;"> Fig 1. Overview of our multispectral object detection framework </div>
</div>

## Installation
```
git clone https://github.com/61s61min/S3Fusion.git
pip install -r requirements.txt
```
## Install Mamba
Please refer to [VMamba](https://github.com/MzeroMiko/VMamba.git)
## Run
You just need to check the path settings in the code to make it run. For more details, please refer to the following [repository](#thanks).

## Dataset
**FLIR**  
 Link:  https://pan.baidu.com/s/19TbR2PDTE3a-b6aG-wzYdA?pwd=q4gb (q4gb)    
**LLVIP**  
 Link:  
**M3FD**  
 Link:  https://github.com/dlut-dimt/TarDAL.git    
**VEDAI**  
 Link:  https://downloads.greyc.fr/vedai/    
**MFNet**  
 Link:  https://pan.baidu.com/s/1QkTA6FTTIMhlwbZfPEjqcA?pwd=1odx (1odx)
**SemanticRT**  
 Link: https://pan.baidu.com/s/13b7FDIO4FxN98xGAliYEIA?pwd=6rl5 (6rl5)
**VT821, VT1000, VT5000**  
 Link: https://pan.baidu.com/s/1Ousl6r2VLpzMowPFOyNRNA?pwd=h0cg (h0cg)
## Weight
**FLIR**  
 Link:  
**LLVIP**  
 Link:  
**M3FD**  
 Link:  
**VEDAI**  
 Link:  
**MFNet**  
 Link:  
**SemanticRT**  
 Link:  
**VT821, VT1000, VT5000**  
 Link: 

## Results：
### Object Detection

<div align="center">
  <img src="OD results.png" width="600px">
  <div style="color:orange; border-bottom: 10px solid #d9d9d9; display: inline-block; color: #999; padding: 10px;"> Fig 1. Overview of our multispectral object detection framework </div>
</div>    

<div align="center">
  <img src="OD.png" width="600px">
  <div style="color:orange; border-bottom: 10px solid #d9d9d9; display: inline-block; color: #999; padding: 10px;"> Fig 1. Overview of our multispectral object detection framework </div>
</div>  

### Semantic Segmentation

<div align="center">
  <img src="SS results.png" width="600px">
  <div style="color:orange; border-bottom: 10px solid #d9d9d9; display: inline-block; color: #999; padding: 10px;"> Fig 1. Overview of our multispectral object detection framework </div>
</div>
<div align="center">
  <img src="semantic segmentation.png" width="600px">
  <div style="color:orange; border-bottom: 10px solid #d9d9d9; display: inline-block; color: #999; padding: 10px;"> Fig 1. Overview of our multispectral object detection framework </div>
</div>

### Salient Object Detection

<div align="center">
  <img src="SOD results.png" width="600px">
  <div style="color:orange; border-bottom: 10px solid #d9d9d9; display: inline-block; color: #999; padding: 10px;"> Fig 1. Overview of our multispectral object detection framework </div>
</div>
<div align="center">
  <img src="SOD compare.png" width="600px">
  <div style="color:orange; border-bottom: 10px solid #d9d9d9; display: inline-block; color: #999; padding: 10px;"> Fig 1. Overview of our multispectral object detection framework </div>
</div>


## SOD Salient Map
link: https://pan.baidu.com/s/1c9ClfwWBoe3o106_OBkyqQ?pwd=r0a5 (r0a5)

## Thanks
Our RGB-T OD code is based on [ICAFusion](https://github.com/chanchanchan97/ICAFusion.git), RGB-T SS code is based on [MFNet](https://github.com/haqishen/MFNet-pytorch.git), and RGB-T SOD code is based on [MSEDNET](https://github.com/Zhou-wy/MSEDNet.git), SOD was evaluated with codes from [PySODMetrics](https://github.com/lartpang/PySODMetrics.git) and their contributions are greatly appreciated!
We will update the code later
