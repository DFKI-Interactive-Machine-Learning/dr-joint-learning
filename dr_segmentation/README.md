**Training**

To train a model, run 

```
python train.py --seed 765 --preprocess '2' --lesion 'EX'
``` 

for training a UNet model to segment Hard Exudates lesion images with the preprocessing method of Contrast Enhancement using random seed 765.

- ``` lesion: MA, HE, EX, SE```

The meaning of each preprocessing index is indicated in the following table.

| Preprocessing Index | Preprocessing Methods |
| :---: | :---: |
| '0' | None |
| '1' | Brightness Balance |
| '2' | Contrast Enhancement |
| '3' | Contrast Enhancement + Brightness Balance |
| '4' | Denoising |
| '5' | Denoising + Brightness Balance |
| '6' | Denoising + Contrast Enhancement |
| '7' | Denoising + Contrast Enhancement + Brightness Balance |


- Some IDRID samples are available in the ```data/IDRID_Samples``` directory
  
- For training, follow the directory structure for your dataset



**🖼️ Evaluation:**


🖼️ Evaluating a Single Image

To run inference on a single image and save both the input and the prediction:

```
python evaluate_single_image.py --image_path 'path_to_image' --model 'path_to_model_weights' --output_dir 'output_directory'
```


🧪 Evaluating the Model on the Test Set
To evaluate a trained model on the full test set:

```
python evaluate_model.py --seed 765 --preprocess '2' --lesion 'EX' --model 'path_to_class_specific_model'
```
This loads the model checkpoint from results/models_ex/model.pth.tar and evaluates it on the EX lesion test set using the same preprocessing method and random seed.



**Shared Model weights for four lesion classes**
| lesion | Model weights Name |
| :---: | :---: |
| 'MA' | unet_model_ma.pth.tar |
| 'HE' | unet_model_he.pth.tar |
| 'SE' | unet_model_se.pth.tar |
| 'EX' | unet_model_ex.pth.tar |


