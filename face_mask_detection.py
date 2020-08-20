import cv2
import argparse
import numpy as np
import pandas as pd
from utils import detector_utils
import datetime


TRAINED_MODEL_DIR = 'frozen_graphs'
PATH_TO_MASK_CKPT = r'C:\Users\prasr\Desktop\detect-person-mask-master\graph\akash mask.pb'
PATH_TO_HUMAN_CKPT =  r'C:\Users\prasr\Desktop\detect-person-mask-master\graph\frozen_inference_graph.pb'

mask_detection_graph, mask_sess = detector_utils.load_inference_graph(PATH_TO_MASK_CKPT)
human_detection_graph, human_sess = detector_utils.load_inference_graph(PATH_TO_HUMAN_CKPT)

if __name__ == '__main__':
	# Detection confidence threshold to draw bounding box
	score_thresh  = 0.80
	#Orientation = 'bt'
	machine_border_perc = float(15)
	safety_border_perc = float(30)
	# Start the video stream
	vs = cv2.VideoCapture(0)

	cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)

	im_height, im_width = None, None
	num_hands_detect = 2
	num_persons = 2
	start_time = datetime.datetime.now()
	num_frames = 0

	try:
		while True:
			rec, frame = vs.read()

			if im_height==None:
				im_height, im_width = frame.shape[:2]

			# Convert image to rgb since opencv loads images in bgr, if not accuracy will decrease.
			try:
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			except:
				print('Error converting to RGB')

			mask_boxes, mask_scores, mask_classes = detector_utils.detect_objects(frame, mask_detection_graph, mask_sess)
			human_boxes, human_scores, human_classes = detector_utils.detect_objects(frame, human_detection_graph, human_sess)

			#safety_position = detector_utils.draw_safety_lines(frame, machine_border_perc, safety_border_perc)
			
			detector_utils.draw_box_on_face(num_hands_detect, score_thresh, mask_scores, mask_boxes, mask_classes, im_width, im_height, frame)
			detector_utils.draw_box_on_person(num_persons, score_thresh, human_scores, human_boxes, human_classes, im_width, im_height, frame)

			num_frames += 1
			elasped_time = (datetime.datetime.now()-start_time).total_seconds()
			fps = num_frames/elasped_time
			cv2.putText(frame, 'FPS: '+str('%0.2f'%(fps)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

			cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

			if cv2.waitKey(1) & 0xFF==ord('q'):
				vs.release()
				cv2.destroyAllWindows()
				break
					
	except:
		pass