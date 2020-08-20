import numpy as np
import sys
import tensorflow as tf
import os
import cv2
import pandas as pd
from utils import label_map_util
from scipy.spatial import distance as dist
#from pygame import mixer



# detection_graph = tf.compat.v1.Graph()

TRAINED_MODEL_DIR = 'frozen_graphs'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT =  r'C:\Users\prasr\Desktop\detect-person-mask-master\graph\akash mask.pb'
PATH_TO_CKPT2 =   r'C:\Users\prasr\Desktop\detect-person-mask-master\graph\frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = TRAINED_MODEL_DIR + '/labelmap.pbtxt'


#NUM_CLASSES = 3
# load label map using utils provided by tensorflow object detection api
#label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
#categories = label_map_util.convert_label_map_to_categories(label_map, 
#											max_num_classes=NUM_CLASSES,
#												use_display_name=True)
#category_index = label_map_util.create_category_index(categories)						

def load_inference_graph(PATH_TO_CKPT):

	print('=======> Loading frozen graph into memory')
	detection_graph = tf.compat.v1.Graph()

	with detection_graph.as_default():
		od_graph_def = tf.compat.v1.GraphDef()
		with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')
			sess = tf.compat.v1.Session(graph=detection_graph)
		print('=======> Detection graph loaded')
		return detection_graph, sess


def detect_objects(image_np, detection_graph, sess):

	image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
	detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
	detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
	detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
	num_detections = detection_graph.get_tensor_by_name('num_detections:0')	

	image_np_expanded = np.expand_dims(image_np, axis=0)

	(boxes, scores, classes, num) = sess.run(
  		[detection_boxes, detection_scores,
    	 detection_classes, num_detections],
    	feed_dict={image_tensor: image_np_expanded})

	return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)


def draw_box_on_face(num_face_detect, score_thresh, scores, boxes, classes, im_width, im_height, image_np):
	
	color = None
	color0 = (0,255,0)
	color1 = (255,0,0)
	color2 = (255,255,0)

	for i in range(num_face_detect):


		if scores[i] > score_thresh:
			item = ''
			if classes[i]==1:
				item = 'With Mask'
				color = color0
			elif classes[i]==2:
				item = 'Without Mask'
				color = color1
			else:
				item = 'Mask Wore Incorrectly'
				color = color2
			
			(x_min, x_max, y_min, y_max) = (boxes[i][1]*im_width, boxes[i][3]*im_width,
											boxes[i][0]*im_height, boxes[i][2]*im_height)

			p1 = (int(x_min), int(y_min))
			p2 = (int(x_max), int(y_max))

			cv2.rectangle(image_np, p1, p2, color, 3, 1)

			cv2.putText(image_np, 'Face '+str(i)+': '+item, (int(x_min), int(y_min)-5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

			cv2.putText(image_np, 'confidence: '+str("{0:.2f}".format(scores[i])),
                       (int(x_min),int(y_min)-20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)




# def draw_safety_lines(image_np, machine_border_perc, safety_border_perc):

# 	posii = int(image_np.shape[1] - image_np.shape[1]/3)

# 	cv2.putText(image_np, 'Blue Line: Machine Border Line', (posii, 30), 
# 		cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,0,255), 1)
# 	cv2.putText(image_np, 'Red Line: Safety Border Line', (posii, 50), 
# 		cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255,0,0), 1)

# 	machine_line_position = int((image_np.shape[0]*machine_border_perc)/100)
# 	safety_position = int((image_np.shape[0]*safety_border_perc)/100)

# 	cv2.line(image_np, (0, machine_line_position), (image_np.shape[1], machine_line_position), (0,0,255), 2, 8)
# 	cv2.line(image_np, (0, safety_position), (image_np.shape[1], safety_position), (255,0,), 2, 8)

# 	return safety_position


def alert_check(image_np, im_width, im_height, p1, p2, point_dict):
	
	# alert_pt = (int(im_height/2), int(im_width/2))
	# mid = p1[0]+(p2[0]-p1[0])//2

	# cv2.line(image_np, (mid, safety_position), (mid, p1[1]), (255,0,0), 1, 8)

	# # mixer.init()
	# # mixer.music.load('utils/alert.wav')

	# if p1[1] <= safety_position:
	# 	cv2.putText(image_np, '[Alert !!!]', alert_pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
	# 	#os.system('utils/alert.wav')
	# 	# mixer.music.play()

	if len(point_dict.items()) > 1:

		if point_dict[0][1][0] < point_dict[1][0][0]:
			c1 = (point_dict[0][1][0], (point_dict[0][0][1]+point_dict[0][1][1])//2)
			c2 = (point_dict[1][0][0], (point_dict[1][0][1]+point_dict[1][1][1])//2)

			cv2.line(image_np, c1, c2, (0,0,255), 2, 8)
			distance = dist.euclidean(c1, c2)
			dist_inch = distance/101.76
			pt = (((c1[0]+c2[0])//2)-10, ((c1[1]+c2[1])//2)-10)
			cv2.putText(image_np, '%0.2f inch'%(dist_inch), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

		elif point_dict[1][1][0] < point_dict[0][0][0]:
			c1 = (point_dict[1][1][0], (point_dict[1][0][1]+point_dict[1][1][1])//2)
			c2 = (point_dict[0][0][0], (point_dict[0][0][1]+point_dict[0][1][1])//2)

			cv2.line(image_np, c1, c2, (0,0,255), 2, 8)
			distance = dist.euclidean(c1, c2)
			dist_inch = distance/101.76
			pt = (((c1[0]+c2[0])//2)-10, ((c1[1]+c2[1])//2)-10)
			cv2.putText(image_np, '%0.2f inch'%(dist_inch), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)





def draw_box_on_person(num_persons, score_thresh, scores, boxes, classes, im_width, im_height, image_np):
	
	color = None
	color0 = (255,255,255)

	point_dict = {}

	for i in range(num_persons):


		if scores[i] > score_thresh:
			item = ''
			if classes[i]==1:
				item = 'Person'
				color = color0
			
			(x_min, x_max, y_min, y_max) = (boxes[i][1]*im_width, boxes[i][3]*im_width,
											boxes[i][0]*im_height, boxes[i][2]*im_height)

			p1 = (int(x_min), int(y_min))
			p2 = (int(x_max), int(y_max))

			cv2.rectangle(image_np, p1, p2, color, 3, 1)

			cv2.putText(image_np, item+str(i), (int(x_min), int(y_min)-5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

			cv2.putText(image_np, 'confidence: '+str("{0:.2f}".format(scores[i])),
                       (int(x_min),int(y_min)-20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

			point_dict[i] = (p1, p2)

			alert_check(image_np, im_width, im_height, p1, p2, point_dict)