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


1. **Install Pipenv:**
   Pipenv is a package manager that combines pip package installations with virtual environments. If you don't have Pipenv installed, you can install it using pip:
   
   ```shell
   pip install pipenv
   ```

2. **Navigate to your project directory:**
   Open your command-line interface and navigate to the root directory of the project.

3. **Set up a virtual environment and install dependencies:**
   Run the following command to create a new virtual environment and install the dependencies specified in your project's Pipfile:
   
   ```shell
   pipenv install
   ```

   This command will create a new virtual environment if one doesn't exist already and install all the packages listed in the Pipfile. Pipenv will also create a Pipfile.lock file that locks the versions of your installed packages for future reproducibility.

4. **Activate the virtual environment:**
   To activate the virtual environment, run the following command:
   
   ```shell
   pipenv shell
   ```

   This will activate the virtual environment <br>
   
   ##### After above steps , start running S5.ipynb


