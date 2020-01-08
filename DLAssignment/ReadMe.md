Dependencies:

- Python3.5
- Pytorch
- torchvision
- opencv
- pycocotools
- xmltodict

For training, we need to prepare data. This is a one time thisng for a given machine with a partical dataset.
```
python3 PCBchecker.py prepData
```

To train:
```
python3 PCBchecker.py train
```

To Test on one image:
create a folder 'model/' in the directory where you have PCBchecker.py
In the 'model/' folder download these two pretrained models:
[***ObjectDetector***](https://1drv.ms/u/s!Au_917wA6i4miiuXax4IPC_vU_pC?e=yFVRRD)
[***Classifier***](https://1drv.ms/u/s!Au_917wA6i4miiy_A0103y14E_Ka?e=8a0lPD)
Finally run the following and pass the path of image to be tested:
```
python3 PCBchecker.py demo <imagePath>
```

Note: The DLAssignment/ folder has just two images and excel with number_to_dict.json file as a sample. Replace this with actual data in the same format. Model was trained on a 1080Ti.

Remarks for improvement: The two networks are separate as of now, One could use a siamese style network to compare or make an auxiliary task to find defects as a Multi Task learning scenario to the object detector with a weighted loss for the auxiliary task.

Outputs for Detector training:
![alt text](https://github.com/LalitPradhan/NanoNetsPCBChecker/blob/master/DLAssignment/detector.png)

Outputs for Classifier training:
![alt text](https://github.com/LalitPradhan/NanoNetsPCBChecker/blob/master/DLAssignment/Classifier.png)
