# Object_Part_Detection
Tensorflow implementation of Fully Convolutional Neural Network to extract features from different parts of human object


## Keypoints of this project

- Fine tune FCN8 toward [PASCAL part dataset](http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/pascal_part.html). 
- Using SLIC algorithm to segment each image into superpixels. 
- Extract deep features from superpixels. 

### Object part segmentation samples below

The first image at the top is the original image. Image at lower left is the result we got, and image at lower right is the ground truth.


<img src="https://user-images.githubusercontent.com/17235054/32005885-670c32da-b973-11e7-992c-e5f4942d95be.jpg" width=300 height=250>
<img src="https://user-images.githubusercontent.com/17235054/32005883-66dbf444-b973-11e7-8faa-1da55d94f099.png" width=700 height=250>


The second example: 

<img src="https://user-images.githubusercontent.com/17235054/32005886-6724d8c6-b973-11e7-9cee-e17613f14bac.jpg" width=300 height=250>
<img src="https://user-images.githubusercontent.com/17235054/32005884-66ec1c20-b973-11e7-9902-6dd0cc27d033.png" width=700 height=250>

The third example that contains multiple human in one image:

<img src="https://user-images.githubusercontent.com/17235054/32005890-6a6f6dd4-b973-11e7-8b74-3cc91a7d3943.jpg" width=300 height=250>
<img src="https://user-images.githubusercontent.com/17235054/32005882-66a22b9c-b973-11e7-8bbc-8423b2038a84.png" width=700 height=250>


### SLIC algorithm implementation sample below

<img src="https://user-images.githubusercontent.com/17235054/32006355-9ee29b1c-b974-11e7-8ac2-56e299552dc2.png" width=900 height=400>

Zoomed in the samples above:

<img src="https://user-images.githubusercontent.com/17235054/32006356-9ef484ee-b974-11e7-9fc5-a2eab91ac429.png" width=900 height=400>

The original code for fully convolutional neural is from Marvinteichmann, please refers to his github page [here](https://github.com/MarvinTeichmann). 
