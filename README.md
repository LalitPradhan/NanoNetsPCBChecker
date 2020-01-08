# NanoNetsPCBChecker
# Deep Learning Assignment
PCB Component Defect Identification. Can you identify defect components on a PCB Board using Machine Learning/Deep Learning?

## Description
There is a PCB Manufacturer, who wants to automatically detects on their PCBs. Components are laid out and there could be two types of defects. All the PCBs are supposed to look identical (with some tolerance for error in placement)
1. Missing Components - No component should be missing
2. Rotated Components - No component should be oriented incorrectly

### Correct PCB
Here is what a correct PCB looks like:
<br>
<img src="https://nanonets-assignments.s3-us-west-2.amazonaws.com/1.jpg" height="300px">

### Error 1 - Missing Component
<br>
<img src="https://nanonets-assignments.s3-us-west-2.amazonaws.com/3.3.jpg" height="300px">

### Error 2 - Incorrectly oriented component
<br>
<img src="https://nanonets-assignments.s3-us-west-2.amazonaws.com/7.7.jpg" height="300px">



## Dataset
You can download dataset from this link
[Dataset](https://nanonets-assignments.s3-us-west-2.amazonaws.com/DLAssignment.zip)

### Dataset Structure
```
DLAssignment
├── 1.1.jpg
├── 1.1.xml
├── 1.jpg
├── 1.xml
├── ...
├── ...
├── errors
│   ├── 1.1.jpg
│   ├── 2.2.jpg
│   ├── .....
├── number_to_component.json
```
Each image (.jpg file) has a corresponding annotation (.xml) file which has locations of each of the components marked on it.

For each image that has a defect on it, there is a image with the same name in the errors folder. The images in the errors folder, have the errors drawn on the image in red.

### Annotation Details
If a PCB has all of the components on it then it's corresponding XML file will have 60 bounding boxes drawn on it. These XML files are in [VOC pascal format](https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5). If a component is missing no annotation would be found in it's place. If the component is rotated, then the field rotated would be marked as ```<rotated>1</rotated>```

**Sample annotation of a single component:**
```
<annotation>
	<folder>ImageSets</folder>
	<filename>1.1.jpg</filename>
	<path>/1.1.jpg</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>3840</width>
		<height>2748</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<!--Lots of other objects here.....-->
	<object>
		<name>1_a</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<rotated>0</rotated>
		<bndbox>
			<xmin>771</xmin>
			<ymin>377</ymin>
			<xmax>1366</xmax>
			<ymax>569</ymax>
		</bndbox>
	</object>
	<!--Lots of other objects here.....-->
</annotation>
```

### Naming convention of each component
There are a total of 60 unique components. There might be components of the same type. A component would be labelled `1_a` or `57_af`, `57` here would represent the unique component number, `af` would represent the component type. Example both `59_o` and `60_o` are of the same type, but have different positions on the PCB. This might be helpful. 

The file ```number_to_component.json``` contains the mapping between the numbers and the types.



## Problem Statement
Write a python function that works in the following way:

#### Run Command:
```
python process.py path_to_image
```

#### Case 1: Errors - Prints:
```
Component 13 Rotated
Component 22 Missing
Component 31 Missing
```


#### Case 2: No Errors - Prints:
```
No Errors
```



## Requirements
1) You need to implement deep learning/machine learning based models to solve this problem (you can use 1, or many).  
2) Identify components that are missing
3) Identify components that are rotated
4) The script should accept an image path as an input and print the output in the specified format
5) Use any code, model, technology, library you want. 
6) **Bonus if your algorithm is extensible to other PCB designs as well, which can be trained given adequate training data in the same format and no other human intervention**



## Deliverables
1) A working script which accepts an image as an input and prints the output as specified. Along with instructions to use and setup.
2)  A github repo with model training code and code for any preprocessing, data analysis. Even jupyter notebook should work. We don't need to be able to run this code, just see the method followed.
3) Accuracy/Other metrics of the model/s
4) A short write up of the approach you used (this could be a doc, a readme, or inline code comments or comments in a jupyter notebook)

### Train Test Validation
1) Split the data for training and validation however you like
2) We have a seperate test set that we will use for testing your code




## Things we will judge you on (in order of importance):
1) Finishing the assignment - Detecting missing components (25%)
1) Finishing the assignment - Detecting rotated components (25%)
2) Approach (25%)
2) Code quality (15%)
3) Accuracy (10%)
4) Bonus (25% extra)


# Use ANY code tools technology and make ANY assumptions needed to complete the assignment on time
