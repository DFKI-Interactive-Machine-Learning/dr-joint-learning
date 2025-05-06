To train a model, run 

```
python train_fgadr.py --seed 765 --preprocess '2' --lesion 'EX'
``` 

for training a UNet model to segment Hard Exudates lesion images with the preprocessing method of Contrast Enhancement using random seed 765.

- ```lesion: MA, HE, EX, SE, SG```, SG for combined segmentation masks

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


🖼️ Evaluating a Single Image
To run inference on a single image and save both the input and the prediction:

```
python evaluate_single_image.py --image_path ./data/sample_image.png --model ./checkpoints/model.pth.tar --output_dir ./results
```
This will:

Resize and save the input image as sample_image_input.png

Predict and save the mask as sample_image_prediction.png
All outputs are stored in the folder specified by --output_dir.

🧪 Evaluating the Model on the Test Set
To evaluate a trained model on the full test set:

```
python evaluate_model.py --seed 765 --preprocess '2' --lesion 'EX' --model results/models_ex/model.pth.tar
```
This loads the model checkpoint from results/models_ex/model.pth.tar and evaluates it on the EX lesion test set using the same preprocessing method and random seed.
