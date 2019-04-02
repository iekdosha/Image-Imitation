# My first Tensorflow project
This script visualizes how a simple network understands an image in grayscale.

before we start lets look at some results then explain them.
for the image:  

![](https://i.imgur.com/pHELj2G.jpg")  

The output images will be:   

<img src="/result.gif?raw=true"> 


Now that we could see a possible output of this script we could explain :)
The methos is as follows (very abstract):
1. load an image in graysacle
2. build a cnn (from config file)
3. for each apoch:  
	3.1 if points refresh is due: refresh random points  
	3.2 feed points into cnn and test how close the output (white-black) to the real image pixel on that points  
	3.3 minimize the loss (difference from real image values)  

output images will be saved where u run this script and config file should be in that folder aswell.

## configuration:
*img_path*: path to the input image

*no_of_random_points*: number of random points to train on for each refresh

*no_of_iteration_before_refresh*: number of iterations before each points refresh

*hidden_layers*: hidden layers (not the input 2 or output 2), example: "hidden_layers": [17,10,17]

*learning_rate*: gradient descent learning rate

*epoch_count*: iteration count

