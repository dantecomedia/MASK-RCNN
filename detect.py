#https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/

from xml.etree import ElementTree
from os import listdir
from  numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from  matplotlib import pyplot 

class KangarooDataset(Dataset):
	def load_dataset(self,dataset_dir,is_train=True):
		self.add_class("dataset",1,"kangaroo")
		dataset_dir="/mnt/F8F8B8AFF8B86E0E/kangaroo-master"
		images_dir=dataset_dir+'/images/'
		annotations_dir=dataset_dir+'/annots/'
		for filename in listdir(images_dir):
			image_id=filename[:-4]
			if image_id in ['00090']:
				continue
			if is_train and int(image_id)>=150:
				continue
			if not is_train and int(image_id) <150:
				continue
			img_path=images_dir+filename
			ann_path=annotations_dir+image_id+'.xml'
			self.add_image('dataset',image_id=image_id,path=img_path,annotation=ann_path)

	def extract_boxes(self,filename):
		tree= ElementTree.parse(filename)
		root=tree.getroot()
		boxes=list()
		for box in root.findall('.//bndbox'):
			xmin=int(box.find('xmin').text)
			ymin=int(box.find('ymin').text)
			xmax=int(box.find('xmax').text)
			ymax=int(box.find('ymax').text)
			coors=[xmin,ymin,xmax,ymax]
			boxes.append(coors)
		width=int(root.find(".//size/width").text)
		height=int(root.find(".//size/height").text)
		return boxes,width,height

	def load_mask(self,image_id):
		info=self.image_info[image_id]
		path=info['annotation']
		boxes,w,h=self.extract_boxes(path)
		masks=zeros([h,w,len(boxes)],dtype='uint8')

		class_ids=list()
		for i in range(len(boxes)):
			box=boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0] , box[2]
			masks[row_s:row_e, col_s:col_e, i]=1
			class_ids.append(self.class_names.index('kangaroo'))
		return masks,asarray(class_ids,dtype='int32')

	def image_reference(self, image_id):
		info=self.image_info[image_id]
		return info['path']

train_set=KangarooDataset()
train_set.load_dataset('kangaroo', is_train=True)
train_set.prepare()
print("Train %d" % len(train_set.image_ids))

test_set=KangarooDataset()
test_set.load_dataset('kangaroo', is_train=False)
test_set.prepare()
print("Test %d" % len(test_set.image_ids))


image_id=0
image=train_set.load_image(image_id)
print(image.shape)


mask,class_ids=train_set.load_mask(image_id)
print(mask.shape)

pyplot.imshow(image)
pyplot.imshow(mask[:,:,0], cmap='gray', alpha=0.5)
pyplot.show()

for i in range(9):
	pyplot.subplot(330+1+i)
	image=train_set.load_image(i)
	pyplot.imshow(image)
	mas,_=train_set.load_mask(i)
	for j in range(mask.shape[2]):
		pyplot.imshow(mask[:,:,j], cmap='gray', alpha=0.3)
pyplot.show()