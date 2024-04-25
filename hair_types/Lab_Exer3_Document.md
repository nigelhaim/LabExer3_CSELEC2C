### # Lab Exercise 3

### Our approach

**Pre-processing part**

**Initial run**
At first we initially run the model from the model and obtain the metrics of accuracy, precision, recall and F1 score. Then made changes to observe how it affects on every change we do. When the metrics are obtain, we revert it back to its original form to observe other changes. When all observation needed was made we note and explain how did the change affect the performance in a technical aspect. We also created a brute force algorithm to observe how each change affects performance in predicting individual pictures and see how close each classification to each other. Each picture is compared to the ground truth and the algorithm also computes how many corrected and incorrect predictions per class. To have more efficient workflow and to further explore all the results, we transfer all of the codes in a python file and run every change in the model. In figure 1 our algorithm on manually checking each image on all classes is shown as:

```
Initialize correct_Predictions dictionary with keys 'Curly_Hair', 'Straight_Hair', 'Wavy_Hair' and values 0
Initialize Incorrect_Predictions dictionary with keys 'Curly_Hair', 'Straight_Hair', 'Wavy_Hair' and values 0
Initialize empty lists all_labels and all_predictions

For each hair_type in the directory 'hair_types':
    Construct hair_type_dir path by joining data_dir with hair_type
    For each filename in the hair_type_dir:
        Construct image_path by joining hair_type_dir with filename
      
        Load and preprocess the image located at image_path
        Convert the image to an array
      
        Make a prediction using the model
        Determine the predicted class index and convert it to a label
      
        Print the predicted class index, the probabilities, predicted label, and actual hair type
      
        Update the counts of correct_Predictions and Incorrect_Predictions based on whether the predicted label matches the actual hair type
      
        Append the actual hair type and predicted label to all_labels and all_predictions lists

```

### Running the model for the first time.

From the file Demo we initially have a Sequential model with three convolution 2D layers each layer will have a stride of one and a padding of valid each layers have a filter of 4, 8, 16 respectively. Every layer contains the kernel size of 20, 8, and 4.  All layers have a padding of 'valid' and a dilation rate of one. Each layer has also the stride of 1The optimizer used for the first time is the Adam Optimizer. The number of Epochs is 50.

### Constant values

The researchers decided that the padding will be constantly 'valid', the dilation rate and the stride the value of 1. Since valid padding has the ability to compute the matrix of valid elements of the image than adding new elements therefore all proportions are accurate. Unlike Casual Padding that is used mainly for forecasting and Same Padding adds elements that could lead inaccurate classifications to the model. Dilation rate is the value of how many pixels are skipped in each step in the convolution therefore the researchers kept it to the value of one as the most minimum number to retain information on every pixel. To maintain constant information the stride is also one therefore no pixel is skipped when the kernel moves across the image.

### Evaluation

The researchers evaluates the model in three ways. First is the computation of Accuracy, precision, Recall and the F1 Score. These methods were computed through the sklearn package. The researchers also developed a brute force algorithm method where a the program loops on the three folders (classes) and run a the predict function of the model on every image in the dataset. Each correct and incorrect prediction is incremented in the list that is based on the Hair type ground truth. If the model predicts the hair type incorrected then the Hair type in the Incorrect Prediction list is incremented. This is done to observe if the model performs Classification bias on a class. The model is also evaluated through a History line Graph to observe if the model overfits. It illustrates on the increasing accuracy and Precision as the Epochs increase with its corresponding loss.

**First run**

| Metric    | Result              |
| --------- | ------------------- |
| Accuracy  | 0.37244897959183676 |
| Precision | 0.3679675705898798  |
| Recall    | 0.37244897959183676 |
| F1 Score  | 0.3698387351663139  |

### Epochs Change

The researchers started to understand how the changes of Epochs affects on how to the model performs. After running the model with only changed of the epochs it was observed that when the epochs is significantly decrease it does not absorb all the necessary information on the training dataset. At the same time too much epochs leading to overfit therefore the model is basically strict or biased on only the features learned from the training dataset and looses its ability to classify new or unseen data. Regardless of the learning the researchers also decided to run the model on different epochs on every change in the model. However on the algorithm of individually predicting the data it is observed that the increasing number of Epochs actually increases the number of correct predictions than incrorrect predictions. This is prone to the concept of overfitting since it memorizes the ground truth rather than learning the features.

### Understanding of Conv2D

In tensorflow Conv2D handles the kernel that applies the filters that handles the channels of the image, and the kernel size to read information. The researchers to use Conv2D than Conv3D since Conv2D focuses on data with different inputs unlike Conv3D it is used to learn on similar frames like a video sequence on every input data.

### MaxPool2D

Applying MaxPool2D downsamples inputs along its spatial dimensions through taking the maximum value on an input. To focus on keeping information the researchers decided to have a constant pool size of 2x2. With the same reason to keeping the kernel size with the value of three, the researchers focuses on keeping optimal information as possible without overfitting. Max Pooling reduces the number of pixels therefore the number of epochs are also reduced.

### Adam Optimizer vs RMSprop vs Stochastic Gradient Descent

The researchers explored three optimizers because of their capabilities stated in numerous studies and articles studied by the researchers. At the demo file given to the researchers Adam Optimizer is used to run the model. The researchers experimented with RMSprop which slightly decreases the performance of the model. This is because despite RMSprop of accelerating optimization process through decreasing the number of function evaluation in order to reach local minimum, it does not have the capabilities of Adam optimizers. On the other hand, Adam optimizers also contain similarities of the RMSProp approach. However the one of the advantages of Adam Optimizer from RMSProp is it calculates uncentered variance of gradients while not subtracting the mean. It extends the capabilities of SGD while having similar approach from RMSProp. These tests were also done with multiple learning rates ranging from 0.1 to 1e-4.

Furthermore despite with the lack of time the researchers attempted but did not optimize the model for the SGD experiments. SGD optimizer significantly decreases the model down to 20%.

The researchers decided to use Adam optimizer as the default optimizer for the model.

![[Pasted image 20240425223252.png]]
*Validation accuracy and loss when SGD is applied*

![[Pasted image 20240425223321.png]]
*Precision when SGD is applied*

### GlobalMaxpool2D vs GlobalAveragePool2D

The researchers ran the model with two Global poolings. GlobalAveragePool2D performs better than GlobalMaxPool2D. It was stated by Zafar et al., (2022) that regardless of importance, Average pooling weights activation equally. Therefore even with other backgrounds and noise in the image, it wont affect the classification ability of the model. Unlike MaxPool2D despite that it captures important features better, it relies on the consistent size, minimal noise, and little to no abnormal features in the image. This indicated on the researchers to further explore preprocessing images. As the researchers developed it is decided to temporarily use GlobalAveragePool2D despite of studying the images with noises and abnormalities to further developed the model.

### Filters

**Switching the filters from Large to Little**
The researchers explored how filters affects the performance. Initially three layers contain 4, 8, 16 of filter values respectively. Then the researchers switched the filter values of the first and third layer. The performance of the metrics significantly dropped and the number of incorrect classifications increases in the Straight and Wavy hair classes. The experiment is also repeated after applying Local Max pooling and an additional layer. The results show similarities. It is therefore discussed that it is because as the image is down sampled, the patterns are getting more complexed therefore it requires more filters to absorb more channels and capture the patterns.

**Doubling the filter value**

After understanding the concept of how filters work. It is then decided to explore the relationship between filters and epochs. Hypothetically if the filters increase therefore less epochs to be needed to save more time and further explore in improving the model. However increasing the number of filters also increases computing time therefore lesser epochs is needed. Despite of increasing and decreasing of filters the researchers managed to only achieve the metrics in Table 1 without any preprocessing and Regularization

| Metric            | Result              |
| ----------------- | ------------------- |
| Accuracy          | 0.39285714285714285 |
| Precision         | 0.39228485044811573 |
| Recall            | 0.39285714285714285 |
| F1 Score          | 0.3925236120508474  |
| ![[Figure_1.png]] |                     |

![[Figure_2.png]]

## Pre Processing

### Image sharpening

Image sharpening enhances the definition of edges of an image. It is done through the comparison of adjacent pixels in an image and enhancing the brightness to emphasize edges of the image. Applying this preprocessing increases all metrics by 0.0100. Theoretically the performance improved also because Max Pooling extracts edges and points which is highlighted by the preprocessing method.
