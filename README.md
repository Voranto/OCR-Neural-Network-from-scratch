# OCR-Neural-Network-from-scratch

# REQUIREMENTS
- Latest Python version: https://www.python.org/downloads/
- Pip, as a python library installer: https://pip.pypa.io/en/stable/cli/pip_download/
- The following libraries, which can be installed through pip:
  - Open the command prompt and paste the following commands:

```
pip install pygame
pip install matplotlib
pip install nnfs
pip install numpy
pip install pandas
```
# EXECUTE PYTHON FILES
Instead of double clicking, open the command prompt in windows, and navigate to the project folder using the "cd" command. For example, if the project folder is situated in "Downloads" and the project folder name is "Neural_network", when opening the command prompt you should paste the following command:
```
cd Downloads
cd Neural_network
```
After you are situated in the correct directory, execute either of the files using the command:
```
python3 FILENAMEHERE.py
```

# TRAINING THE NEURAL NETWORK
This project contains two files "main.py", which is the one in charge of training the neural network itself and saving its weights and biases into csv files. There are already some presets installed in the folders "test_final" and "test_final2". To train new data, executing "main.py" will be needed, and you may adjust parameters such as layer composition, momentum, decay, activation functions, optimizers (SGD is currently used as a preset) and learning rate. To test the neural network previously trained, paste all the csv created after "main.py" has been executed into "test_final2", and delete all the previous data in that folder.

# TRYING THE NEURAL NETWORK
To test the neural network, you are offered two options: either draw you own numbers, or run it through a test of 10000 images and checking its accuracy. If drawing, the percentages on the right hand side of the pygame window represent 

Some advice when drawing your own numbers:
- Try to center the number as much as possible, both with its position and its angle.
- Try to avoid having the number fill the whole grid, as the training numbers dont
