# ELG 5271 Artificial Intelligence for Cybersecurity Applications -- Project -- Group 12
==========================================================================================

## Malware Detection using PNG images ##

This project is about classifying malware classes from given image dataset. The dataset contains 28 classes and are given in the format 3D numpy array and RGB csv file. The given dataset is not trainable format nor cleansed data. Hence, cleaning needs to be done before proceeding it for training.

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

Dataset for both traditional and deep learning are splitted into `train` and `test` respectively. The split ratio is `[70:30]`. 




