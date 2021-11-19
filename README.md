# PFLD Tensorflow 2

Implementation of PFLD A Practical Facial Landmark Detector by Tensorflow 2.

## Usage
1. Requirements: tensorflow >= 2.0.0, numpy, opencv

2. Datasets

   > WFLW Dataset Download
   >
   >  Wider Facial Landmarks in-the-wild (WFLW) is a new proposed face dataset. It contains 10000 faces (7500 for training and 2500 for testing) with 98 fully manual annotated landmarks.
   >
   > 
   >
   > WFLW Training and Testing images [Google Drive](https://drive.google.com/file/d/1hzBd48JIdWTJSsATBEB_eFVvPL1bx6UC/view?usp=sharing) [Baidu Drive](https://pan.baidu.com/s/1paoOpusuyafHY154lqXYrA)
   >
   > [WFLW Face Annotations](https://wywu.github.io/projects/LAB/support/WFLW_annotations.tar.gz)
   >
   > Unzip above two packages and put them on ./data/WFLW/
   >
   > move Mirror98.txt to WFLW/WFLW_annotations
   >
   > $ cd data 
   >
   > $ python3 SetPreparation.py

3. Train

   You can change configurations in `train.py` and `config.py`. For training, just execute one line code.

    ```shell
    python train.py
    ```

4. Test

   Just read `test.py` and load weight you want.


## Others
1. For loss function, `attributes_w_n` may all be zero, which makes loss equal to zero. So it may need to rethink about the weight.

2. We also provide a model called `PFLD_wing_loss_fn` which uses wing_loss and removes auxiliarynet. 

## Reference
1. [PFLD-pytorch](https://github.com/polarisZhao/PFLD-pytorch)
