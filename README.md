# MATH3001
Project Title: "Deep Learning for Medical Image Classification".

Teacher: Dr Luisa Cutillo.

For Section 6 of my project, I will be investigating the Malaria Dataset found at: https://lhncbc.nlm.nih.gov/publication/pub9932.

The set consists of 27588 cell images, aqcuired by light microscopes attached to smartphone cameras. Half of the images are malaria infected cells and the other half are uninfected. The pre-processing of the data set is done by following Dr Adrian Rosebrocks' tutorial shared at: https://www.pyimagesearch.com/2018/12/03/deep-learning-and-medical-image-analysis-with-keras/. The script build_dataset.py from the tutorial is used to build the dataset. The resulting dataset is broken down into 3 folders: training, testing and validation. This format requires further modification in order to be compatible with the customised models. Namely, the folders are manually broken down to Infected_train, Uninfected_train, Infected_test and Uninfected_test. The validation set will not be used since the model where validation is required has a built-in command for doing this.

The images are classified using two several methods. That is, classification is performed using both a fully connected neural network and a convolutional neural network. 

* The data is imported by customising the first part of the Dogs vs Cats classifier found at: https://pythonprogramming.net/loading-custom-data-deep-learning-python-tensorflow-keras/

* The training of the fully connected neural network (Model 1) is done by customising the Fashion MNIST classifier found at: https://www.tensorflow.org/tutorials/keras/classification

* The training of the CNN (Model 2) is done by custimising the Dogs vs Cats Classifier found at: https://pythonprogramming.net/convolutional-neural-network-deep-learning-python-tensorflow-keras/?completed=/loading-custom-data-deep-learning-python-tensorflow-keras/

* Data visualisation is again performed by customising the MNIST classifier.
After applying both models on the Malaria dataset, Model 2 will be further adjusted in order to classify X-Ray images of healthy lungs vs COVID-19 Infected lunngs. The dataset is shared by Adrian Rosebrock at: https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/.

The COVID-19 Dataset, built by Rosebrock consists of 25 images of healthy lungs found at: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia and 25 images of infected lungs originally found at: https://github.com/ieee8023/covid-chestxray-dataset/tree/master/images. 
