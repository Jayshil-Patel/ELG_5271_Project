# ELG 5271 Artificial Intelligence for Cybersecurity Applications -- Project -- Group 12
===========================================================================

## Malware Detection using PNG images ##

This project is about classifying malware classes from given image dataset. The dataset contains 28 classes and are given in the format 3D numpy array and RGB csv file. The given dataset is not trainable format nor cleansed data. Hence, cleaning needs to be done before proceeding it for training.

The recommended environment to run the codes it `Google Colab` with `GPU` runtime. If you are running on your local computers, install the necessary packages by running the command `pip3 install -r requirements.txt`. 

--------------------------------------------------------------------------

Summary of Code files:
1. Data Pre-Processing:
[dataset.ipynb](https://github.com/Jayshil-Patel/ELG_5271_Project/blob/master/dataset.ipynb) is for creating image dataset used for training deep learning models.\
[DataPreprocessing.ipynb](https://github.com/Jayshil-Patel/ELG_5271_Project/blob/master/DataPreprocessing.ipynb) is for creating dataset for training machine learning models.\
[fol.py](https://github.com/Jayshil-Patel/ELG_5271_Project/blob/master/fol.py) is for splitting image dataset into train and test folders.

2. Deep Learning Models:
[tranfer_learning_ensemble.ipynb](https://github.com/Jayshil-Patel/ELG_5271_Project/blob/master/tranfer_learning_ensemble.ipynb) is to run the deep learning models (VGG, DenseNet, ResNet).\
[visualization.ipynb](https://github.com/Jayshil-Patel/ELG_5271_Project/blob/master/visualization.ipynb) is to visualize hidden layers

3. Machine Learning Models and Ensemble:
[MalDetection.ipynb](https://github.com/Jayshil-Patel/ELG_5271_Project/blob/master/MalDetection.ipynb) is to run the machine learning models and ensemble models.

So, the main trianing and test result model codes are: [tranfer_learning_ensemble.ipynb](https://github.com/Jayshil-Patel/ELG_5271_Project/blob/master/tranfer_learning_ensemble.ipynb) and [MalDetection.ipynb](https://github.com/Jayshil-Patel/ELG_5271_Project/blob/master/MalDetection.ipynb)


_The following steps were performed to make the dataset trainable:_

### Step 1: Data Reading ###

The data is read using `pandas` package. Few of the dataset rows were `bad_lines` hence, `error_bad_lines=False` function was used to skip the badlines. 

```python
dataframe = pd.read_csv(filePath, index_col=False, warn_bad_lines=True, error_bad_lines=False)
```

### Step 2: Data Transformation ###

1.  The next step is to transform the data from the CSV file to PNG images. First, the labels file is concatenated with the main image data array CSV and the label column is used for creating the folders to store the images into their respective class folders. 

2. Each row is converted into numpy format, and reshaped into the given image size `(224, 224)`. 

3. _For Deep Learning Models:_ Each array is then stored in `.png` format as an image into their respective class label folders. This image format is used for training deep learning models, and splitted into `train` and `test` folders. 

_Label folder creation:_
```python
import os
main_path = "*image path/"
for each in range(0,len(labels_img)):
    print(labels_img.iloc[each,0].astype(int))
    path = main_path+str(labels_img.iloc[each,0].astype(int))
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        print("already exists")
```

_Images creation:_
```python
for i, (x, y) in enumerate(zip(labels_img.values, data_img.values)):
    print(i, x.astype(int), y)
    mat = np.resize(y,(224,224))
    img = Image.fromarray(np.uint8(mat * 255) , 'L')
    img = img.convert('RGB')
    img.save("path/"+str(x[0].astype(int))+"/"+str(i)+".png")
    img = 0
```

4. _For Traditional ML Models:_ For traditional ML models, the dataset is split into `train X`, `train y`, `test X`, `test y`. The arrays are stored in `pickle` file to load the dataset for training ML models. 

_Dataset creation:_
```python
dataset2 = []
for i in range(len(images.values)):
    img_vector = images.values[i][0].split(',')
    img = np.resize(img_vector, (224, 224))
    dataset2.append(img.astype(float))
dataset2 = np.array(dataset2)
labels = [label if label < 10 else label - 1 for label in labels] # Since there is no class 10, shift all the labels grater than 10
labels = np.array(labels)
print('shape of the dataset: ' + str(dataset2.shape))
print('shape of the labels: ' + str(labels.shape))
# save the results into a pickle file:
save_obj(dataset2, 'cleaned_dataset')
save_obj(labels, 'cleaned_dataset_labels')
```

### Step 3: Data Splitting ###

Dataset for both traditional and deep learning are splitted into `train` and `test` respectively. The split ratio is `[80:20]`. Run the following code to split your data:
[fol.py](https://github.com/Jayshil-Patel/ELG_5271_Project/blob/ashish/fol.py). Change your dataset paths by giving input images and output directory to save the splitted data. The dataset should be in the format:

```
input/
    class1/
        img1.jpg
        img2.jpg
        ...
    class2/
        imgWhatever.jpg
        ...
    ...
```

```python
import splitfolders
splitfolders.ratio("input_images/, 
    output="splitted_folders/",
    seed=1337, ratio=(.8, .2), group_prefix=None, move=False)
```
*Finally! The data is now ready for training!!*
--------------------------------------------------------------------------

*Now, training the models using Transfer Learning*

Transfer Learning models with pre-trained `ImageNet` weights are used to train the model. The base layers are froze and last layers are activated with `Softmax` activation function with `28` output layers. 

Go to `Google Colab`, and save the `runtime` as `GPU`. Upload this notbook [tranfer_learning_ensemble.ipynb](https://github.com/Jayshil-Patel/ELG_5271_Project/blob/ashish/tranfer_learning_ensemble.ipynb). Now, use the refined dataset to train the model [dataset link](https://drive.google.com/file/d/1awPqXYmxEnRN1bRtoKYYZ9Brm1Q7MDke/view?usp=sharing). Now change the train and test paths as mentioned below:

```python
DIR_TRAIN = '/content/drive/MyDrive/gray_mal_dataset/train/'
DIR_TEST = '/content/drive/MyDrive/gray_mal_dataset/val/'
```

Now, run each cell in the notebook. 

### Model 1: VGG16 ###

```python
model_vgg19_bn = models.vgg19_bn(pretrained=True)
for param in model_vgg19_bn.parameters():
    param.requires_grad = False
model_vgg19_bn.classifier[6] = torch.nn.Linear(in_features=model_vgg19_bn.classifier[6].in_features, out_features=28)
model_vgg19_bn = model_vgg19_bn.to(DEVICE)
```

![VGG 16 training accuracy graph](https://github.com/Jayshil-Patel/ELG_5271_Project/blob/ashish/graphs/vgg16_graph.png)

### Model 2: ResNet152 ###

```python
model_resnet152 = models.resnet152(pretrained=True)
for param in model_resnet152.parameters():
    param.requires_grad = False
model_resnet152.fc = torch.nn.Linear(model_resnet152.fc.in_features, 28)
model_resnet152 = model_resnet152.to(DEVICE)
```

![ResNet152 training accuracy graph](https://github.com/Jayshil-Patel/ELG_5271_Project/blob/ashish/graphs/resnet152_graph.png)

### Model 3: DenseNet161 ###

```python
model_densenet161 = models.densenet161(pretrained=True)
for param in model_densenet161.parameters():
    param.requires_grad = False
model_densenet161.classifier = torch.nn.Linear(model_densenet161.classifier.in_features, out_features=28)
model_densenet161 = model_densenet161.to(DEVICE)
```

![DenseNet161 training accuracy graph](https://github.com/Jayshil-Patel/ELG_5271_Project/blob/ashish/graphs/densenet161_graph.png)

The models are trained individually and stored as weight files `.pth`. 

*Deep Learning Models are ready!*
--------------------------------------------------------------------------

*Now, time for Machine Learning Models*

For the machine learning models, use this cleaned dataset to train the ML Models. [dataset link](https://drive.google.com/file/d/1ykFi8YPKYsJH7dxahgViAd9MzfJJ5a1D/view?usp=share_link).

Go to the jupyter notbook [MalDetection.ipynb](https://github.com/Jayshil-Patel/ELG_5271_Project/blob/Hamed/MalDetection.ipynb) and change the dataset path as mentioned below:

```python
dir = 'G:/University/Ph.D/AI for Sybersecurity/Project/Cleaned Datasets/'
```

Now, run each cell to get the ML model results. 

### 5-fold cross validation ###

The dataset is computed with 5-fold cross validation for training with each model and the best of 3 models were taken for ensemble model. 

```python
kf = KFold(n_splits=5, shuffle=True, random_state=12)
```
The ML models used were 5, `SVM`, `KNN`, `Logistic Regression`, `Random Forest`, `Decision Tree`. Of these 5 models, highest accuracies were noted for `Logistic Regression`, `SVM`, `Random Forest` models. Then, hyper-parameter tuning is used for to choose the best resultant hyperparameters. 

```python
trainX = load_obj('train_lbp_featurs')
trainY = load_obj('trainY')

param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
svc=svm.SVC(probability=True)
model=GridSearchCV(svc,param_grid)
model.fit(trainX, trainY)
print('Best Score : ', model.best_score_)
print('Best Parameters : ', model.best_params_)
```

### Feature Extraction using Local Binary Pattern ###

LBP technique is used for feature extraction. The following code snippet describes the execution of the LBP technique of the dataset. 

```python
lbp_features = []
num_points = 100
radius = 8
eps=1e-7
for img in test_dataset:
    lbp = feature.local_binary_pattern(img, num_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    lbp_features.append(hist)    
lbp_features = np.array(lbp_features)
print('shape of the resulted feature vectors: ' + str(lbp_features.shape))   
# save the result into a pickle file:
save_obj(lbp_features, 'test_lbp_featurs')
```
Now, load the final dataset with the feature extraction function:

```python
trainX = load_obj('final_train_lbp_featurs') 
testX = load_obj('final_test_lbp_featurs')
trainY = load_obj('final_train_labels')
testY = load_obj('final_test_labels')
```

### Training ML Models ###

Now, everything is pre-processed and we can train the ML models. 

_Random Forest Model_

```python
model = RandomForestClassifier(n_estimators=100, random_state = 12)
model.fit(trainX, trainY)
predicted_RF = model.predict(testX)
accuracy = metrics.accuracy_score(testY, predicted_RF)
print(accuracy)
save_obj(predicted_RF, 'predicted_RF')
```

_SVM Model_

```python
model = svm.SVC(kernel='rbf', C=0.1, gamma= 0.0001)
model.fit(trainX, trainY)
predicted_SVM = model.predict(testX)
accuracy = metrics.accuracy_score(testY, predicted_SVM)
print(accuracy)
save_obj(predicted_SVM, 'predicted_SVM')
```

_Logistic Regression Model_

```python
model = LogisticRegression()
model.fit(trainX, trainY)
predicted_LR = model.predict(testX)
accuracy = metrics.accuracy_score(testY, predicted_LR)
print(accuracy)
save_obj(predicted_LR, 'predicted_LR')
```

*ML Models are ready!*
--------------------------------------------------------------------------
### Ensemble Models ###

After the deep learning and ML models are trained and the trained weights and variables are executed. Now, we will combine the trained models for performing ensemble model. 

Load the predictions of each deep learning model in CSV file:

```python
predicted_densenet = np.genfromtxt(dir + 'densenet_test_predictions_ensemble.csv', delimiter=',')
predicted_resnet = np.genfromtxt(dir + 'resnet152_test_predictions_ensemble.csv', delimiter=',')
predicted_vgg = np.genfromtxt(dir + 'vgg_test_predictions_ensemble.csv', delimiter=',')
```

Now, execute the cells to perform ensemble model. 

```python
combined_deep = np.array([predicted_densenet, predicted_resnet, predicted_vgg])
majority_vote1 = mode(combined_deep)[0][0]
accuracy_deep = metrics.accuracy_score(testY, majority_vote1)

# ensemble the results of the all selected traditional ML models using majority voting:
combined_tred = np.array([predicted_RF, predicted_LR, predicted_SVM])
majority_vote2 = mode(combined_tred)[0][0]
accuracy_tred = metrics.accuracy_score(testY, majority_vote2)

# ensemble the results of all the models using majority voting:
combined_tred = np.array([predicted_RF, predicted_LR, predicted_SVM, predicted_densenet, predicted_resnet, predicted_vgg])
majority_vote3 = mode(combined_tred)[0][0]
accuracy_all = metrics.accuracy_score(testY, majority_vote3)

print('the accuracy when combining deep models: ', accuracy_deep)
print('the accuracy when combining traditional ML models: ', accuracy_tred)
print('the accuracy when combining all the models: ', accuracy_all)
```

The above code snippet performs the ensemble model for ML models, DL models and for all the 6 models. 

Here are the confusion matrix plots for ensemble models:

_Ensemble model of DL models:_

![Confusion Matrix of Ensemble model of DL models](https://github.com/Jayshil-Patel/ELG_5271_Project/blob/ashish/graphs/confusion_matrix_ensemble_dl.png)

_Ensemble model of ML models:_

![Confusion Matrix of Ensemble model of ML models](https://github.com/Jayshil-Patel/ELG_5271_Project/blob/ashish/graphs/confusion_matrix_ensemble_ml.png)

_Ensemble model of all models:_

![Confusion Matrix of Ensemble model of ML models](https://github.com/Jayshil-Patel/ELG_5271_Project/blob/ashish/graphs/confusion_matrix_ensemble_all_models.png)

*Ensemble Models are ready!*
--------------------------------------------------------------------------

Now, out of curiosity, we even visualized hidden layers of an given image with ResNet model. Go to the notebook [visualization.ipynb](https://github.com/Jayshil-Patel/ELG_5271_Project/blob/ashish/visualization.ipynb). Give an input image in the following line:

```python
image = Image.open(str('/content/257.png'))
plt.imshow(image)
```

Execute each cell in the notebook and you will get a visualization of the hidden layers. 

![Hidden Layers of ResNet](https://github.com/Jayshil-Patel/ELG_5271_Project/blob/ashish/graphs/hidden_layers.png)

We hope you enjoyed and get to know more about our project
--------------------------------------------------------------------------
--------------------------------------------------------------------------

# ELG 5271 Artificial Intelligence for Cybersecurity Applications -- Project -- Group 12

--------------------------------------------------------------------------
--------------------------------------------------------------------------
