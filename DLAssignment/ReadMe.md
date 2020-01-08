Dependencies:

- Python3.5
- Pytorch
- torchvision
- opencv
- pycocotools

For training, we need to prepare data. This is a one time thisng for a given machine with a partical dataset.
```
$python3 PCBchecker.py prepData
```

To train:
```
$python3 PCBchecker.py train
```

To Test on one image:
create a folder 'model/' in the directory where you have PCBchecker.py
In the 'model/' folder download these two pretrained models:
[***ObjectDetector***](https://1drv.ms/u/s!Au_917wA6i4miiuXax4IPC_vU_pC?e=yFVRRD)
[***Classifier***](https://1drv.ms/u/s!Au_917wA6i4miiy_A0103y14E_Ka?e=8a0lPD)
Finally run the following and pass the path of image to be tested:
```
$python3 PCBchecker.py demo <imagePath>
```


