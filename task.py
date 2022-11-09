import cv2
import numpy as np

def load_object():
       value = cv2.dnn.readNetFromDarknet("cfg","weights")
       classes = []
       with open ("E:\syed_task\coco.name","r") as f:
           classes = [ lines.strip() for lines in f.readlines()]
           layer_name = value.getLayerNames()
           output_layer = [layer_name[i - 1] for i in value.getUnconnectedOutLayers()]
           colors = np.random.uniform(0, 255, size = (len(classes),3))
           return value,classes,colors,output_layer


     
def load_image(load_object):
	img = cv2.imread("image")
	img = cv2.resize(img, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape
	return img, height, width, channels


def detect_objects(load_image,img, value, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	value.setInput(blob)
	outputs = value.forward(outputLayers)
	return blob, outputs



def get_box_dimensions(detect_objects,outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
	     for detect in output:
                     scores = detect[5:]
                     print(scores)
                     class_id = np.argmax(scores)
                     conf = scores[class_id]
                     if conf > 0.3:
                        center_x = int(detect[0] * width)
                        center_y = int(detect[1] * height)
                        w = int(detect[2] * width)
                        h = int(detect[3] * height)
                        x = int(center_x - w/2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confs.append(float(conf))
                        class_ids.append(class_id)
                        return boxes, confs, class_ids


                    
def draw_labels(get_box_dimensions,boxes, confs, colors, class_ids, classes, img): 
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			color = colors[i]
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
	cv2.imshow("Image", img)          
 

 
def start_video(draw_labels):
    model, classes, colors, output_layers = load_object()
    video_path = cv2.VideoCapture("videos.mp4")
    while True:
                    frame = video_path.read[1]
                    height, width = frame.shape()
                    blob, outputs = detect_objects(frame, model, output_layers)
                    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
                    draw_labels(boxes, confs, colors, class_ids, classes, frame)
                    key = cv2.waitKey(1)
                    if key == 27:
                        break
    video_path.release()
    
    
start_video(draw_labels)    
    
            
