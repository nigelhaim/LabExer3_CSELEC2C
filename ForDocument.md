# For the Document

### Made an algorithm where it tests the prediction and computes the total accuracy and precision
- Measured the accuracy, precision, recall, and F1

### Epochs change without MaxPool2D()

1 Epoch
Accuracy: 0.29591836734693877
Precision: 0.1881546269301371
Recall: 0.29591836734693877
F1 Score: 0.14918890633176346

50 Epochs

Accuracy: 0.37244897959183676
Precision: 0.3679675705898798
Recall: 0.37244897959183676
F1 Score: 0.3698387351663139

75 Epochs
Accuracy: 0.2857142857142857
Precision: 0.08205128205128205
Recall: 0.2857142857142857
F1 Score: 0.12749003984063745

Reason: 
 your neural network may overfit, meaning that it will memorize the training data and lose its ability to generalize to new and unseen data

### Epochs change with MaxPool2D

50 Epochs

Accuracy: 0.3469387755102041
Precision: 0.21843342193389495
Recall: 0.3469387755102041
F1 Score: 0.25543516844789826

Accuracy: 0.30612244897959184
Precision: 0.29926890296348424
Recall: 0.30612244897959184
F1 Score: 0.29962980448207066


### Also made a manual for loop for each image to observe how does the model predicts correctly and incorrectly
- This can be inaccurate with the initial metrics applied in the model 


### First and 3rd Layer filters are flipped without MaxPool2D()
Accuracy: 0.2857142857142857
Precision: 0.08163265306122448
Recall: 0.2857142857142857
F1 Score: 0.12698412698412698