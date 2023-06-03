# This Repo has CNN model template

### Files in the Repo	Description

| File Name  | Description |
| ------------- | ------------- |
| model.py  | this file as the model definition  |
| utils.py  | this file as the Utility functions to train,test model  |
| S5.ipynb  | this is the main file ,which will run the model and inference |

### Model Architecture 

|  Layer (type)           |    Output Shape       |  Parameters  |
| ----------------------- | --------------------- | --------- |
|      Conv2d-1           | [-1, 32, 26, 26]        |     320 |
|      Conv2d-2           | [-1, 64, 24, 24]         | 18,496 |
|      Conv2d-3           | [-1, 128, 10, 10]         | 73,856 |
|      Conv2d-4            | [-1, 256, 8, 8]         |295,168 |
|      Linear-5             |      [-1, 50]        | 204,850 |
|      Linear-6              |     [-1, 10]          |   510 |


Total params: 593,200 <br/>
Trainable params: 593,200 <br/>
Non-trainable params: 0 <br/>

## How to use 


