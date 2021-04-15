# grasp_prediction
Python 2, conda, pytorch, and some other libraries imported in the code are also needed
Python 3 is also working, but may need to modify a bit, I did tried to run with Python 3 but not with this version directly

To run the code, go to the python folder:
python run_grasp.py 
The default parameters for number of epochs etc can be found in run_grasp.py, can easily change while running
The data loader is in graspdata.py, only right, or left or both can be loaded. 'both2' refers to concatenate left and right, so the input to the network is 6xHxW.
If use 'both2', the network in the models.py, the input_channels = 6, otherwise = 3.

The results will be saved in logs/tmp if not set to specific location

The results can be visualized by running plot_logGrasp.py

In the graspdata.py, I did not do any data augmentaion, but only normalization

Data are in the data folder.
