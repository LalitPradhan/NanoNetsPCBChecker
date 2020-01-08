import torch
from PIL import Image
import torchvision
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from engine import train_one_epoch, evaluate
import utils
import random
import shutil
import xmltodict
import glob
import cv2
import copy
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from torchvision import transforms
import time

dataPath = 'DLAssignment/'
datapathAugmented = 'AugmentedData/'
dataClfPath = 'classificationData/'
if not os.path.exists(dataClfPath):
	os.mkdir(dataClfPath)
# dataClfPath = 'classificationData/
modelsPath = 'model/'
if not os.path.exists(modelsPath):
	os.mkdir(modelsPath)
# datapathAugmentedMarked = 'AugmentedDataMarked/'
pad = 200 # num pixels outside tight bounding box of detected object to be considered for crop for classification
detecterModelTrained = modelsPath+'Augdetector_final.pth'
classifierModelTrained = modelsPath+'classifier_best.pth'
num_epochs_detector = 100
num_epochs_classifier = 100
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def augmentLocData(dataPath):
	print ('Augmenting Given data ...')
	files = sorted(glob.glob(dataPath+'/*jpg'))
	fxmls = sorted(glob.glob(dataPath+'/*xml'))
	if not os.path.exists(datapathAugmented):
		os.mkdir(datapathAugmented)
	# if not os.path.exists(datapathAugmented):
	# 	os .mkdir('AugmentedDataMarked/')
	
	print ('\tSaving Original data ...')
	for xml in fxmls:
		with open(xml) as fd:
			image = cv2.imread(xml.split('xml')[0]+'jpg')
			data = xmltodict.parse(fd.read(), process_namespaces=True)
			gTruth = {}
			# imgCopy1 = copy.deepcopy(image)
			for obNum in range(len(data['annotation']['object'])):
				xmin = int(data['annotation']['object'][obNum]['bndbox']['xmin'])
				xmax = int(data['annotation']['object'][obNum]['bndbox']['xmax'])
				ymin = int(data['annotation']['object'][obNum]['bndbox']['ymin'])
				ymax = int(data['annotation']['object'][obNum]['bndbox']['ymax'])
				name = data['annotation']['object'][obNum]['name']
				gTruth[name]=[xmin, ymin, xmax, ymax]
				rot = data['annotation']['object'][obNum]['rotated']
				imgCopy = copy.deepcopy(image)
				imgCopy = cv2.rectangle(imgCopy, (xmin,ymin), (xmax,ymax), (0,0,255), 2)
				snip = imgCopy[ymin-pad:ymax+pad,xmin-pad:xmax+pad,:]
				fname = dataClfPath+'Original_'+xml.split('/')[-1].split('.xml')[0]+ '_'+ name + '_' + str(rot) + '.jpg'
				cv2.imwrite(fname, snip)
				# imgCopy = cv2.rectangle(imgCopy1, (xmin,ymin), (xmax,ymax), (0,0,255), 2)
		# fname1 = datapathAugmentedMarked+'Original_'+xml.split('/')[-1].split('xml')[0]+'jpg'
		# cv2.imwrite(fname1, imgCopy1)
		imgCopy2 = copy.deepcopy(image)
		fname2 = datapathAugmented+'Original_'+xml.split('/')[-1].split('xml')[0]+'jpg'
		cv2.imwrite(fname2, imgCopy2)
		with open(fname2.split('jpg')[0]+'json', 'w') as outfile:
			json.dump(gTruth, outfile)

	print ('\tSaving Horizontally flipped data ...')
	for xml in fxmls:
		with open(xml) as fd:
			image = cv2.imread(xml.split('xml')[0]+'jpg')
			image = cv2.flip(image, 1)
			data = xmltodict.parse(fd.read(), process_namespaces=True)
			gTruth = {}
			# imgCopy1 = copy.deepcopy(image)
			for obNum in range(len(data['annotation']['object'])):
				xmin = abs(image.shape[1] - 1 - int(data['annotation']['object'][obNum]['bndbox']['xmin']))
				xmax = abs(image.shape[1] - 1 - int(data['annotation']['object'][obNum]['bndbox']['xmax']))
				ymin = int(data['annotation']['object'][obNum]['bndbox']['ymin'])
				ymax = int(data['annotation']['object'][obNum]['bndbox']['ymax'])
				name = data['annotation']['object'][obNum]['name']
				gTruth[name]=[xmax, ymin, xmin, ymax]
				rot = data['annotation']['object'][obNum]['rotated']
				imgCopy = copy.deepcopy(image)
				imgCopy = cv2.rectangle(imgCopy, (xmin,ymin), (xmax,ymax), (0,0,255), 2)
				snip = imgCopy[ymin-pad:ymax+pad,xmax-pad:xmin+pad,:]
				fname = dataClfPath+'Hflip_'+xml.split('/')[-1].split('.xml')[0]+ '_'+ name + '_' + str(rot) + '.jpg'
				cv2.imwrite(fname, snip)
				# imgCopy = cv2.rectangle(imgCopy1, (xmin,ymin), (xmax,ymax), (0,0,255), 2)			
		# fname1 = datapathAugmentedMarked+'Hflip_'+xml.split('/')[-1].split('xml')[0]+'jpg'
		# cv2.imwrite(fname1, imgCopy1)
		imgCopy2 = copy.deepcopy(image)
		fname2 = datapathAugmented+'Hflip_'+xml.split('/')[-1].split('xml')[0]+'jpg'
		cv2.imwrite(fname2, imgCopy2)
		with open(fname2.split('jpg')[0]+'json', 'w') as outfile:
			json.dump(gTruth, outfile)

	print ('\tSaving Vertically flipped data ...')
	for xml in fxmls:
		with open(xml) as fd:
			image = cv2.imread(xml.split('xml')[0]+'jpg')
			image = cv2.flip(image, 0)
			data = xmltodict.parse(fd.read(), process_namespaces=True)
			gTruth = {}
			# imgCopy1 = copy.deepcopy(image)
			for obNum in range(len(data['annotation']['object'])):
				xmin = int(data['annotation']['object'][obNum]['bndbox']['xmin'])
				xmax = int(data['annotation']['object'][obNum]['bndbox']['xmax'])
				ymin = abs(image.shape[0] - 1 - int(data['annotation']['object'][obNum]['bndbox']['ymin']))
				ymax = abs(image.shape[0] - 1 - int(data['annotation']['object'][obNum]['bndbox']['ymax']))
				name = data['annotation']['object'][obNum]['name']
				gTruth[name]=[xmin, ymax, xmax, ymin]
				rot = data['annotation']['object'][obNum]['rotated']
				imgCopy = copy.deepcopy(image)
				imgCopy = cv2.rectangle(imgCopy, (xmin,ymin), (xmax,ymax), (0,0,255), 2)
				snip = imgCopy[ymax-pad:ymin+pad,xmin-pad:xmax+pad,:]
				fname = dataClfPath+'Vflip_'+xml.split('/')[-1].split('.xml')[0]+ '_'+ name + '_' + str(rot) + '.jpg'
				cv2.imwrite(fname, snip)
				# imgCopy = cv2.rectangle(imgCopy1, (xmin,ymin), (xmax,ymax), (0,0,255), 2)
		# fname1 = datapathAugmentedMarked+'Vflip_'+xml.split('/')[-1].split('xml')[0]+'jpg'
		# cv2.imwrite(fname1, imgCopy1)
		imgCopy2 = copy.deepcopy(image)
		fname2 = datapathAugmented+'Vflip_'+xml.split('/')[-1].split('xml')[0]+'jpg'
		cv2.imwrite(fname2, imgCopy2)
		with open(fname2.split('jpg')[0]+'json', 'w') as outfile:
			json.dump(gTruth, outfile)

	print ('\tSaving Diagonally flipped data ...')
	for xml in fxmls:
		with open(xml) as fd:
			image = cv2.imread(xml.split('xml')[0]+'jpg')
			image = cv2.flip(image, -1)
			data = xmltodict.parse(fd.read(), process_namespaces=True)
			gTruth = {}
			# imgCopy1 = copy.deepcopy(image)
			for obNum in range(len(data['annotation']['object'])):
				xmin = abs(image.shape[1] - 1 - int(data['annotation']['object'][obNum]['bndbox']['xmin']))
				xmax = abs(image.shape[1] - 1 - int(data['annotation']['object'][obNum]['bndbox']['xmax']))
				ymin = abs(image.shape[0] - 1 - int(data['annotation']['object'][obNum]['bndbox']['ymin']))
				ymax = abs(image.shape[0] - 1 - int(data['annotation']['object'][obNum]['bndbox']['ymax']))
				name = data['annotation']['object'][obNum]['name']
				gTruth[name]=[xmax, ymax, xmin, ymin]
				rot = data['annotation']['object'][obNum]['rotated']
				imgCopy = copy.deepcopy(image)
				imgCopy = cv2.rectangle(imgCopy, (xmin,ymin), (xmax,ymax), (0,0,255), 2)
				snip = imgCopy[ymax-pad:ymin+pad,xmax-pad:xmin+pad,:]
				fname = dataClfPath+'HVflip_'+xml.split('/')[-1].split('.xml')[0]+ '_'+ name + '_' + str(rot) + '.jpg'
				cv2.imwrite(fname, snip)
		# 		imgCopy = cv2.rectangle(imgCopy1, (xmin,ymin), (xmax,ymax), (0,0,255), 2)				
		# fname1 = datapathAugmentedMarked+'HVflip_'+xml.split('/')[-1].split('xml')[0]+'jpg'
		# cv2.imwrite(fname1, imgCopy1)
		imgCopy2 = copy.deepcopy(image)
		fname2 = datapathAugmented+'HVflip_'+xml.split('/')[-1].split('xml')[0]+'jpg'
		cv2.imwrite(fname2, imgCopy2)
		with open(fname2.split('jpg')[0]+'json', 'w') as outfile:
			json.dump(gTruth, outfile)

	print ('\tSaving Rotated 90 anticlockwise data ...')
	for xml in fxmls:
		with open(xml) as fd:
			image = cv2.imread(xml.split('xml')[0]+'jpg')
			image = np.rot90(image)
			image = np.ascontiguousarray(image, dtype=np.uint8)
			data = xmltodict.parse(fd.read(), process_namespaces=True)
			gTruth = {}
			# imgCopy1 = copy.deepcopy(image)
			for obNum in range(len(data['annotation']['object'])):
				xmin = abs(image.shape[0] - 1 - int(data['annotation']['object'][obNum]['bndbox']['xmin']))
				xmax = abs(image.shape[0] - 1 - int(data['annotation']['object'][obNum]['bndbox']['xmax']))
				ymin = int(data['annotation']['object'][obNum]['bndbox']['ymin'])
				ymax = int(data['annotation']['object'][obNum]['bndbox']['ymax'])
				name = data['annotation']['object'][obNum]['name']
				gTruth[name]=[ymin, xmax, ymax, xmin]
				rot = data['annotation']['object'][obNum]['rotated']
				imgCopy = copy.deepcopy(image)
				imgCopy = cv2.rectangle(imgCopy, (ymin,xmin), (ymax,xmax), (0,0,255), 2)
				snip = imgCopy[xmax-pad:xmin+pad,ymin-pad:ymax+pad,:]
				fname = dataClfPath+'Rot90_'+xml.split('/')[-1].split('.xml')[0]+ '_'+ name + '_' + str(rot) + '.jpg'
				cv2.imwrite(fname, snip)
		# 		imgCopy = cv2.rectangle(imgCopy1, (ymin,xmin), (ymax,xmax), (0,0,255), 2)
		# fname1 = datapathAugmentedMarked+'Rot90_'+xml.split('/')[-1].split('xml')[0]+'jpg'
		# cv2.imwrite(fname1, imgCopy1)
		imgCopy2 = copy.deepcopy(image)
		fname2 = datapathAugmented+'Rot90_'+xml.split('/')[-1].split('xml')[0]+'jpg'
		cv2.imwrite(fname2, imgCopy2)
		with open(fname2.split('jpg')[0]+'json', 'w') as outfile:
			json.dump(gTruth, outfile)

	print ('\tSaving Rotated 90 anticlockwise  with Horizontal flip data ...')
	for xml in fxmls:
		with open(xml) as fd:
			image = cv2.imread(xml.split('xml')[0]+'jpg')
			image = np.rot90(image)
			image = cv2.flip(image, 1)
			image = np.ascontiguousarray(image, dtype=np.uint8)
			data = xmltodict.parse(fd.read(), process_namespaces=True)
			gTruth = {}
			# imgCopy1 = copy.deepcopy(image)
			for obNum in range(len(data['annotation']['object'])):
				xmin = abs(image.shape[0] - 1 - int(data['annotation']['object'][obNum]['bndbox']['xmin']))
				xmax = abs(image.shape[0] - 1 - int(data['annotation']['object'][obNum]['bndbox']['xmax']))
				ymin = abs(image.shape[1] - 1 - int(data['annotation']['object'][obNum]['bndbox']['ymin']))
				ymax = abs(image.shape[1] - 1 - int(data['annotation']['object'][obNum]['bndbox']['ymax']))
				name = data['annotation']['object'][obNum]['name']
				gTruth[name]=[ymax, xmax, ymin, xmin]
				rot = data['annotation']['object'][obNum]['rotated']
				imgCopy = copy.deepcopy(image)
				imgCopy = cv2.rectangle(imgCopy, (ymin,xmin), (ymax,xmax), (0,0,255), 2)
				snip = imgCopy[xmax-pad:xmin+pad,ymax-pad:ymin+pad,:]
				fname = dataClfPath+'Rot90Hflip_'+xml.split('/')[-1].split('.xml')[0]+ '_'+ name + '_' + str(rot) + '.jpg'
				cv2.imwrite(fname, snip)
		# 		imgCopy = cv2.rectangle(imgCopy1, (ymin,xmin), (ymax,xmax), (0,0,255), 2)
		# fname1 = datapathAugmentedMarked+'Rot90HFlip_'+xml.split('/')[-1].split('xml')[0]+'jpg'
		# cv2.imwrite(fname1, imgCopy1)
		imgCopy2 = copy.deepcopy(image)
		fname2 = datapathAugmented+'Rot90HFlip_'+xml.split('/')[-1].split('xml')[0]+'jpg'
		cv2.imwrite(fname2, imgCopy2)
		with open(fname2.split('jpg')[0]+'json', 'w') as outfile:
			json.dump(gTruth, outfile)

	print ('\tSaving Rotated 90 anticlockwise  with Vertical flip data ...')
	for xml in fxmls:
		with open(xml) as fd:
			image = cv2.imread(xml.split('xml')[0]+'jpg')
			image = np.rot90(image)
			image = cv2.flip(image, 0)
			image = np.ascontiguousarray(image, dtype=np.uint8)
			data = xmltodict.parse(fd.read(), process_namespaces=True)
			gTruth = {}
			# imgCopy1 = copy.deepcopy(image)
			for obNum in range(len(data['annotation']['object'])):
				xmin = abs(image.shape[0] - 1 - abs(image.shape[0] - 1 - int(data['annotation']['object'][obNum]['bndbox']['xmin'])))
				xmax = abs(image.shape[0] - 1 - abs(image.shape[0] - 1 - int(data['annotation']['object'][obNum]['bndbox']['xmax'])))
				ymin = int(data['annotation']['object'][obNum]['bndbox']['ymin'])
				ymax = int(data['annotation']['object'][obNum]['bndbox']['ymax'])
				name = data['annotation']['object'][obNum]['name']
				gTruth[name]=[ymin, xmin, ymax, xmax]
				rot = data['annotation']['object'][obNum]['rotated']
				imgCopy = copy.deepcopy(image)
				imgCopy = cv2.rectangle(imgCopy, (ymin,xmin), (ymax,xmax), (0,0,255), 2)
				snip = imgCopy[xmin-pad:xmax+pad,ymin-pad:ymax+pad,:]
				fname = dataClfPath+'Rot90Vflip_'+xml.split('/')[-1].split('.xml')[0]+ '_'+ name + '_' + str(rot) + '.jpg'
				cv2.imwrite(fname, snip)
		# 		imgCopy = cv2.rectangle(imgCopy1, (ymin,xmin), (ymax,xmax), (0,0,255), 2)
		# fname1 = datapathAugmentedMarked+'Rot90VFlip_'+xml.split('/')[-1].split('xml')[0]+'jpg'
		# cv2.imwrite(fname1, imgCopy1)
		imgCopy2 = copy.deepcopy(image)
		fname2 = datapathAugmented+'Rot90VFlip_'+xml.split('/')[-1].split('xml')[0]+'jpg'
		cv2.imwrite(fname2, imgCopy2)
		with open(fname2.split('jpg')[0]+'json', 'w') as outfile:
			json.dump(gTruth, outfile)

	print ('Data preparation finished.')
	print ('Checking for any corrupt files ...')
	allFiles = sorted(glob.glob(datapathAugmented+'*jpg'))
	for eachFile in allFiles:
		arr = cv2.imread(eachFile)
		if arr is None:
			os.remove(eachFile)

	allFiles = sorted(glob.glob(dataClfPath+'*jpg'))
	for eachFile in allFiles:
		arr = cv2.imread(eachFile)
		if arr is None:
			os.remove(eachFile)
	print ('All checks complete and data prepared.')


class pcbAugDataset(torch.utils.data.Dataset):
	def __init__(self, datapath):
		self.datapath = datapath
		self.imgs = sorted(glob.glob(self.datapath+'/*jpg'))
		self.masks = sorted(glob.glob(self.datapath+'/*json'))		
		with open(dataPath+'number_to_type.json') as json_file:
			jsonData = json.load(json_file)
			self.classNames = sorted(set(jsonData.values()))
			self.classMap = {}
			self.reverseMap = {}
			for classNum, className in enumerate(self.classNames):
				self.classMap[className]=classNum
			self.reverseMap = {v: k for k, v in self.classMap.items()}

	def __getitem__(self, idx):
		img_path = self.imgs[idx]
		mask_path = self.masks[idx]
		img = Image.open(img_path).convert("RGB")			   
		boxes = []
		labels = []
		with open(mask_path) as fd:
			data = json.load(fd)
			for key, value in data.items():
				xmin = value[0]
				xmax = value[2]
				ymin = value[1]
				ymax = value[3]
				boxes.append([xmin, ymin, xmax, ymax])
				labels.append(self.classMap[key.split('_')[-1]])				
		labels = torch.LongTensor(labels)
		boxes = torch.as_tensor(boxes, dtype=torch.float32)
		image_id = torch.tensor([idx])
		area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
		iscrowd = torch.zeros((len(labels),), dtype=torch.int64)		
		target = {}
		target["boxes"] = boxes
		target["labels"] = labels
		target["image_id"] = image_id
		target["area"] = area
		target["iscrowd"] = iscrowd
		img = F.to_tensor(img)
		return img, target

	def __len__(self):
		return len(self.imgs)

class pcbClfDataset(torch.utils.data.Dataset):
	def __init__(self, datapath, transforms=None):
		self.datapath = datapath
		self.transforms = transforms
		self.imgsAll = sorted(glob.glob(self.datapath+'/*jpg'))
		self.imgsRotated = [x for x in self.imgsAll if x.split('_')[-1].split('.jpg')[0]=='1']
		self.imgsUnRotated = [x for x in self.imgsAll if x.split('_')[-1].split('.jpg')[0]=='0']
		if len(self.imgsUnRotated)>len(self.imgsRotated):
			random.shuffle(self.imgsUnRotated)
			self.imgs = self.imgsRotated+self.imgsUnRotated[:3*len(self.imgsRotated)]
		else:
			random.shuffle(self.imgsRotated)
			self.imgs = self.imgsUnRotated+self.imgsRotated[:3*len(self.imgsUnRotated)]

		
	def __getitem__(self, idx):
		img_path = self.imgs[idx]
		img = Image.open(img_path).convert("RGB")
		with open(dataPath+'/number_to_type.json') as json_file:
			jsonData = json.load(json_file)
			self.compList = jsonData.keys()
		nameSplit = img_path.split('_')
		orgFileName = nameSplit[1]+'.jpg'
		compNum = nameSplit[2]
		compType = nameSplit[3]
		augType = nameSplit[0]
		rotType = nameSplit[-1].split('.jpg')[0]
		if int(rotType)==0:
			label = int(compNum) - 1
		else:
			label = int(compNum) - 1 + len(self.compList)
		if self.transforms is not None:
			img = self.transforms(img)
		label = torch.LongTensor([label])
		return img, label

	def __len__(self):
		return len(self.imgs)

def objectDetectorArgs(train=True):
	if train:
		model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
		json_file = open(dataPath+'number_to_type.json')
		num_classes = len(sorted(set(json.load(json_file).values()))) + 1 # 1 class (person) + background
		in_features = model.roi_heads.box_predictor.cls_score.in_features
		model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
		model.to(device)
		params = [p for p in model.parameters() if p.requires_grad]
		optimizer = torch.optim.SGD(params, lr=0.0005,momentum=0.9, weight_decay=0.0005)
		lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3, gamma=0.1)
		num_epochs = num_epochs_detector
		return model, optimizer, lr_scheduler, num_epochs
	else:
		model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
		json_file = open(dataPath+'/number_to_type.json')
		num_classes = len(sorted(set(json.load(json_file).values()))) + 1 # 1 class (person) + background
		in_features = model.roi_heads.box_predictor.cls_score.in_features
		model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
		model.load_state_dict(torch.load(detecterModelTrained))
		model.to(device)
		model.eval()
		return model

def TrainObjectDetector():
	model, optimizer, lr_scheduler, num_epochs = objectDetectorArgs(train=True)
	dataset = pcbAugDataset(datapathAugmented)
	indices = torch.randperm(len(dataset)).tolist()
	dataset_train = dataset#torch.utils.data.Subset(dataset, indices[:-100])
	dataset_val = dataset#torch.utils.data.Subset(dataset, indices[-100:])

	image_datasets = {'train':dataset_train, 'val':dataset_val}
	dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=2, shuffle=True, num_workers=4,collate_fn=utils.collate_fn) for x in ['train', 'val']}
	
	since = time.time()
	for epoch in range(num_epochs):
		train_one_epoch(model, optimizer, dataloaders['train'], device, epoch, print_freq=10)
		if epoch%25==0:
			torch.save(model.state_dict(), modelsPath+'Augdetector_'+str(epoch)+'.pth')
		lr_scheduler.step()
		evaluate(model, dataloaders['val'], device=device)
	torch.save(model.state_dict(), modelsPath+'Augdetector_final.pth')
	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	return model

def objectClassifierArgs(train=True):
	if train:
		# model_ft = models.resnet18(pretrained=True)
		# num_ftrs = model_ft.fc.in_features
		model_ft = models.vgg16_bn(pretrained=True)
		num_ftrs = model_ft.classifier[6].in_features
		with open(dataPath+'/number_to_type.json') as json_file:
			jsonData = json.load(json_file)
			compList = jsonData.keys()
		# model_ft.fc = torch.nn.Linear(num_ftrs, len(compList)*2)
		model_ft.classifier[6] = torch.nn.Linear(num_ftrs, len(compList)*2)
		model_ft = model_ft.to(device)
		criterion = torch.nn.CrossEntropyLoss()
		optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
		exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
		num_epochs = num_epochs_classifier
		return model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs
	else:
		# model_ft = torchvision.models.resnet18(pretrained=False)
		# num_ftrs = model_ft.fc.in_features
		model_ft = models.vgg16_bn(pretrained=True)
		num_ftrs = model_ft.classifier[6].in_features
		with open(dataPath+'/number_to_type.json') as json_file:
			jsonData = json.load(json_file)
			compList = jsonData.keys()
		compList = jsonData.keys()
		# model_ft.fc = torch.nn.Linear(num_ftrs, len(compList)*2)
		model_ft.classifier[6] = torch.nn.Linear(num_ftrs, len(compList)*2)
		model_ft.load_state_dict(torch.load(classifierModelTrained))
		model_ft = model_ft.to(device)
		model_ft.eval()
		return model_ft

def TrainClassifier():
	model, criterion, optimizer, scheduler, num_epochs= objectClassifierArgs(train=True)
	data_transforms = transforms.Compose([
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
	dataset = pcbClfDataset(dataClfPath, data_transforms)
	indices = torch.randperm(len(dataset)).tolist()
	dataset_train = torch.utils.data.Subset(dataset, indices[:-50])
	dataset_val = torch.utils.data.Subset(dataset, indices[-50:])
	
	image_datasets = {'train':dataset_train, 'val':dataset_val}
	dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4,collate_fn=utils.collate_fn) for x in ['train', 'val']}
	dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

	since = time.time()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)
		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()
			else:
				model.eval()
			running_loss = 0.0
			running_corrects = 0
			for inputs, labels in dataloaders[phase]:
				inputs = list(img.to(device) for img in inputs)
				inputs = torch.stack(inputs)
				labels = torch.stack(labels).squeeze()
				labels = labels.to(device)
				optimizer.zero_grad()
				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs)
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)
					if phase == 'train':
						loss.backward()
						optimizer.step()
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)
			if phase == 'train':
				scheduler.step()
			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects.double() / dataset_sizes[phase]
			print('{} Loss: {:.4f} Acc: {:.4f}'.format(
				phase, epoch_loss, epoch_acc))
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())
		if epoch%10==0:
			torch.save(model.state_dict(), modelsPath+'classifier_'+str(epoch)+'.pth')
		torch.save(best_model_wts, modelsPath+'classifier_best.pth')
		print()
	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))
	return model
	
def demo(imgPath):
	json_file = open(dataPath+'number_to_type.json')
	jsonData = json.load(json_file)
	classNames = sorted(set(jsonData.values()))
	classMap = {}
	reverseMap = {}
	for classNum, className in enumerate(classNames):
		classMap[className]=classNum
	reverseMap = {v: k for k, v in classMap.items()}

	img = Image.open(imgPath).convert("RGB")
	imgArr = cv2.imread(imgPath)
	img = F.to_tensor(img)
	image = [img.to(device)]
	detectorModel = objectDetectorArgs(train=False)
	classifierModel = objectClassifierArgs(train=False)
	data_transforms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
	detectorOutputs = detectorModel(image)
	finalOutputs = {}
	componentsFound = []
	imgArrFirstCopy = copy.deepcopy(imgArr)
	for opNum, score in enumerate(detectorOutputs[0]['scores']):
		if score.item()>=0.45:
			label = reverseMap[detectorOutputs[0]['labels'][opNum].item()]
			xmin, ymin, xmax, ymax = detectorOutputs[0]['boxes'][opNum].cpu().data.numpy().astype(np.uint16)
			x1, y1, x2, xy = xmin, ymin, xmax, ymax
			imgArrCopy = copy.deepcopy(imgArr)
			imgArrCopy = cv2.rectangle(imgArrCopy, (xmin,ymin), (xmax,ymax), (0,0,255), 2)

			snip = imgArrCopy[ymin-200:ymax+200,xmin-200:xmax+200,:]
			snip = Image.fromarray(snip[:,:,::-1])
			snip = data_transforms(snip)
			snip = torch.stack([snip.to(device)])
			outputs_ft = classifierModel(snip)
			_, preds = torch.max(outputs_ft, 1)

			if preds.item()<len(jsonData.keys()):
				compNum = preds.item()+1
				rot = 0
				rotText = ' '
			else:
				compNum = preds.item() - len(jsonData.keys()) +1
				rot = 1
				rotText = 'Rotated'

			imgArrFirstCopy = cv2.rectangle(imgArrFirstCopy, (xmin,ymin), (xmax,ymax), (0,0,255), 2)
			font = cv2.FONT_HERSHEY_SIMPLEX
			(text_width, text_height) = cv2.getTextSize(str(compNum)+rotText, font, fontScale=1, thickness=2)[0]
			text_offset_x = x1
			text_offset_y = y1
			box_coords = ((int(round(text_offset_x)), int(round(text_offset_y-10))), (int(round(text_offset_x + text_width - 2)), int(round(text_offset_y - text_height - 2 - 10))))
			imgArrFirstCopy = cv2.rectangle(imgArrFirstCopy, box_coords[0], box_coords[1], (0,0,255), cv2.FILLED)
			imgArrFirstCopy = cv2.putText(imgArrFirstCopy,str(compNum)+rotText,(int(round(x1)),int(round(y1-10))), font, 1,(0,0,0),2,cv2.LINE_AA)
			# assert jsonData[str(compNum)] == label
			if compNum not in componentsFound:
				componentsFound.append(compNum)
				finalOutputs[str(compNum)] = rot
			# fname = 'output/'+str(compNum)+'_'+label+'.jpg'
			# cv2.imwrite(fname, imgArrCopy)
	defects = {}
	allComponents = jsonData.keys()
	for key in allComponents:
		if key in finalOutputs.keys():
			if finalOutputs[key] == 1:
				defects[key]='Rotated'
		else:
			defects[key]='Missing'

	if len(defects.keys())==0:
		print('No Errors')
	else:
		for key, value in defects.items():
			print('Component', key, value)
	cv2.namedWindow("Output", cv2.WINDOW_NORMAL) 
	cv2.imshow('Output', imgArrFirstCopy)
	cv2.waitKey(-1)
	# print(detectorOutputs[0]['scores'])

if __name__ == '__main__':

	if sys.argv[1] == 'prepData':
		augmentLocData(dataPath)
	elif sys.argv[1] == 'train':
		print ('Training Object detector')
		TrainObjectDetector()
		print ('Training Classifier')
		TrainClassifier()
	elif sys.argv[1] == 'demo':
		# demo(dataPath+'1.1.jpg')
		demo(sys.argv[2])
