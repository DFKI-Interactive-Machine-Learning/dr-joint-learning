
## Official Repository for DRG-Net

Tusfiqur, Hasan Md, Duy MH Nguyen, Mai TN Truong, Triet A. Nguyen, Binh T. Nguyen, Michael Barz, Hans-Juergen Profitlich, et al. "DRG-Net: interactive joint learning of multi-lesion segmentation and classification for diabetic retinopathy grading. [arXiv preprint](https://arxiv.org/abs/2212.14615)
- Contains the training scripts for classification and segmentation models for FGADR and EyPACS datasets
- Scripts for Additional models and other datasets will be added soon.


Contact: Hasan Md Tusfiqur Alam (hasan.alam@dfki.de)  
Licenced under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1)

## Installation

Recommended environment:
- python 3.8+
- pytorch 1.7.1+
- torchvision 0.8.2+
- tqdm
- munch
- packaging
- tensorboard
- scikit-learn
- opencv-python
- pillow < 7
- segmentation-models-pytorch

To install the dependencies create a virtualenv and run:
```shell
$ pip install -r requirements.txt
```
## Datasets

Datasets used in this paper can be accessed from the following links:

- [Indian Diabetic Retinopathy Image Dataset - IDRiD](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)
- [FGADR Dataset - Look Deeper into Eyes.](https://csyizhou.github.io/FGADR/)
- [EyePACS Dataset](https://paperswithcode.com/dataset/kaggle-eyepacs)


## For classification

- The 'dr_classification' folder contains scripts for the dr grading classification task.
- Follow readme.md inside 'dr_classification' for instructions.


## For segmentation

- The 'dr_segmentation' folder contains scripts for the dr segmentation task.
-  Follow readme.md inside 'dr_segmentaion' instructions


## Citation
If you use this code or results in your research, please cite:
```
@article{tusfiqur2022drg,
  title={DRG-Net: interactive joint learning of multi-lesion segmentation and classification for diabetic retinopathy grading},
  author={Tusfiqur, Hasan Md and Nguyen, Duy MH and Truong, Mai TN and Nguyen, Triet A and Nguyen, Binh T and Barz, Michael and Profitlich, Hans-Juergen and Than, Ngoc TT and Le, Ngan and Xie, Pengtao and others},
  journal={arXiv preprint arXiv:2212.14615},
  year={2022}
}
```
