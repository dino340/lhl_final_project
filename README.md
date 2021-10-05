# Lighthouse Labs Capsone Project
## Neural Network Facilitated Facemask Detection

Over the last two years facemasks have become a regular part of life for many people, there is strong evidence showing they prevent the spread of respiratory illnesses, and many places have made them mandatory for the time being.

In order for them to be effective they need to not only be worn but also worn properly covering the mouth, nose and chin.

I decided to create and train a model that classifies images of people based on their mask status to differentiate which are wearing masks properly, improperly or not at all.

This application was created by Garrett Brezsnyak for my capstone project for Lighthouse Labs Data Science Program.

It utilizes a CNN model based on VGG16 trained using images from the FFHQ dataset and from MaskedFace-Net.

## Video demo of model in action
![mask classifier gif](https://user-images.githubusercontent.com/30278033/136102353-bf83a7f8-19ac-494d-a2bd-087e9610da75.gif)

## Metrics for final version of model
### Confusion Matrix
![confusionmatrix](https://user-images.githubusercontent.com/30278033/136100821-753dc9b4-52e8-4b8f-ac6f-ee24e63da075.png)
### Model Accuracy
![model_acc](https://user-images.githubusercontent.com/30278033/136100833-3fe37747-87af-4c2d-8ccd-837d80db0b21.png)
### Model Loss
![model_loss](https://user-images.githubusercontent.com/30278033/136100889-4961a70d-985b-44f6-b3ce-71e51f815fe6.png)


## Running code locally
Required packages are listed in requirements.txt along with the versions that I was running while working on this.

If you would like to run the code locally feel free to clone this repo and run the notebooks in Jupyter Lab, instructions on how to organize the image files is detailed in the "Data Preparation.ipynb" file. Be aware training this model requires a fair amount of harddrive space and horsepower, the images take up approximately 115gb of space, and the model was trained using tensorflow utilizing a GTX1070ti graphics card's CUDA cores and training time was approximately 1 hour per epoch.

"OpenCV.py" can also be run independently as long as requirements are met, "Model-004.model" is unzipped to the working directory, and "haarcascade_frontalface_default.xml" is also located in the working directory. Press ESC to stop the webcam feed and exit the program.

Included is also a streamlit app that can be run so long as the requirements are met, "Model-004.model" is unzipped to the working directory, and "haarcascade_frontalface_default.xml" is also located in the working directory. Navigate to the working directory via commandline and launch the app via ```streamlit run streamlit_app.py```

## Data Sources

FFHQ dataset: https://www.kaggle.com/arnaud58/flickrfaceshq-dataset-ffhq

MaskedFace-Net: https://github.com/cabani/MaskedFace-Net

## Model Link

Trained model is avalible on google drive at: https://drive.google.com/file/d/1tTydBREVIZ_hdTdaNtNYw36dcKRRjrz_/view?usp=sharing
