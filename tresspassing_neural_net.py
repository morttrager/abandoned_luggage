
# coding: utf-8

# In[ ]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


# In[ ]:


# MODEL_NAME = 'faster_rcnn_resnet101_coco_2018_01_28'
MODEL_NAME = 'inception'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = 'mscoco_label_map.pbtxt'
NUM_CLASSES = 90


# In[ ]:


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# In[ ]:


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# In[ ]:


def run_inference_for_single_image(image, graph, cls):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}

            for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                      tensor_name)
            if 'detection_masks' in tensor_dict:
              # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                  detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                  tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                  detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate

            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
            cls_b_p = []
            if cls == 'bag':

                for i in range(0,len(output_dict['detection_classes'])):
                        
                    if (int(output_dict['detection_classes'][i])) in [ 27, 31, 33]:

                        ymin = int(output_dict['detection_boxes'][i][0]*image.shape[0])
                        xmin = int(output_dict['detection_boxes'][i][1]*image.shape[1])
                        ymax = int(output_dict['detection_boxes'][i][2]*image.shape[0])
                        xmax = int(output_dict['detection_boxes'][i][3]*image.shape[1])
                        if (xmin == 0 and ymin == 0 and xmax == 0 and ymax == 0):
                            continue
                        else:
                            cls_b_p.append((xmin, ymin, xmax, ymax))
            elif cls == 'person':
                
                for i in range(0,len(output_dict['detection_classes'])):
                       
                    if (int(output_dict['detection_classes'][i])) ==  1:
                        
                        ymin = int(output_dict['detection_boxes'][i][0]*image.shape[0])
                        xmin = int(output_dict['detection_boxes'][i][1]*image.shape[1])
                        ymax = int(output_dict['detection_boxes'][i][2]*image.shape[0])
                        xmax = int(output_dict['detection_boxes'][i][3]*image.shape[1])
                        
                        if (xmin == 0 and ymin == 0 and xmax == 0 and ymax == 0):
                            continue
                        else:
                            cls_b_p.append((xmin, ymin, xmax, ymax))
                    
    return cls_b_p


# In[ ]:


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


# In[ ]:


def iou_rem(bags):
    
    list_bag = list(bags)

    if len(list_bag)>1:
        for i in range(0, len(bags)-1):
            if bags[i] not in list_bag:
                continue
            else:
                for j in range(i+1, len(bags)):
                    iou = bb_intersection_over_union(bags[i],bags[j])
                    if iou>0.70:                  
                        list_bag.remove(bags[j])

    return list_bag


# In[ ]:


from collections import deque
prev_dq = dq=deque(maxlen=150)

def Unattended_object_detection(image):

    global flag
    global prev_dq
    global cur_bags
   
    counter = 0
    
    bags = run_inference_for_single_image(image, detection_graph, cls = 'bag')
    bags = iou_rem(bags)

    for bag in bags:
        
        xmin = bag[0]
        ymin = bag[1]     
        xmax = bag[2]
        ymax = bag[3]
        
        if xmin<= 80:
            xmin = 0
        else:
            xmin -= 80

        if ymin<= 90:
            ymin = 0
        else:
            ymin -= 90

        if xmax + 80 >= image.shape[1]:
            xmax = image.shape[1]
        else:
            xmax += 80

        if ymax + 90 >= image.shape[0]:
            ymax = image.shape[0]
        else:
            ymax += 90
       
       
        persons = run_inference_for_single_image( image[ymin : ymax, xmin : xmax ] , detection_graph, cls = 'person')
        cnt = 0
       
        
        if len(persons) is 0:
            for l in range(0,len(prev_dq)):
                if bag in prev_dq[l]:
                    cnt +=1
                else:
                    
                    for j in range(0,len(prev_dq[l])): 
                        iou = bb_intersection_over_union(bag,prev_dq[l][j])
                        if iou > 0.60:
                            cnt+=1
                            break
            if cnt <= 30:
                cv2.rectangle(image, (bag[0], bag[1]), (bag[2], bag[3]), (0,0,255), 2)
                cv2.putText(image,"Unattended Lugagge ", (bag[0],bag[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0))
            elif 30 < cnt <= 120:
                cv2.rectangle(image, (bag[0], bag[1]), (bag[2], bag[3]), (0,255,0), 2)
                cv2.putText(image,"Warning Please Check", (bag[0],bag[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,165,0))
            else:
                cv2.rectangle(image, (bag[0], bag[1]), (bag[2], bag[3]), (255,0,0), 2)
                cv2.putText(image,"Abandoned Luggage", (bag[0],bag[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
           
        else:
            counter +=1
    
    if counter ==  len(bags):
        cv2.putText(image,"Normal", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,100,0))
    
    prev_dq.append(bags)
    
    return image


# In[ ]:


from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[ ]:


white_output = 'test123_solution.mp4'
clip1 = VideoFileClip("test123.mp4")
 
# Process video
white_clip = clip1.fl_image(Unattended_object_detection) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'white_clip.write_videofile(white_output, audio=False)')

