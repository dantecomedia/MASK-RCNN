
#https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/



#extracting the bounding boxes


from xml.etree import ElementTree

def extract_bounding_boxes(filename):
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


boxes,width,height=extract_bounding_boxes("/mnt/F8F8B8AFF8B86E0E/kangaroo-master/annots/00002.xml")
print(boxes,width,height)

class Kangroo(dataset):
	def load_dataset(self,dataset_dir,is_train=True):



	def load_mask(self, image_id):



	def image_reference(self, image_id):
