#Mon 11 May 2020 04:26:00 PM +0545 
#Sanjeev Poudel
#Real time Face Detection and Capturing using opencv dnn
#Nepali Videos #Haribahadur #Sali_Man_Paryo


import cv2, time

#load model
model_path = 'models/opencv_face_detector_uint8.pb'
config_path = 'models/opencv_face_detector.pbtxt'

net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
conf_threshold = 0.7

# initialize video source, default 0 (webcam)
video_path = 'videos/haribahadur.mp4'
#video_path = 'videos/sali.mp4'
capture = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('%s_output_opencv_dnn.mp4' % (video_path.split('.')[0]), fourcc, capture.get(cv2.CAP_PROP_FPS), (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))

frame_count, tt = 0, 0

while capture.isOpened():
  ret, img = capture.read()
  if not ret:
    break

  frame_count += 1

  start_time = time.time()

#prepare input
  result_img = img.copy()
  h, w, _ = result_img.shape
  blob = cv2.dnn.blobFromImage(result_img, 2.0, (300, 300), [104, 117, 123], False, False)
  net.setInput(blob)

#inference, find faces
  detections = net.forward()

#postprocessing
  for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > conf_threshold:
      x1 = int(detections[0, 0, i, 3] * w)
      y1 = int(detections[0, 0, i, 4] * h)
      x2 = int(detections[0, 0, i, 5] * w)
      y2 = int(detections[0, 0, i, 6] * h)

      # draw rectangles
      cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 0), int(round(h/150)), cv2.LINE_AA)
      cv2.putText(result_img, '%.2f%%' % (confidence * 100.), (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 50),2, cv2.LINE_AA)

#inference time
  tt += time.time() - start_time
  fps = frame_count / tt
  cv2.putText(result_img, 'FPS frame_rate(dnn): %.2f' % (fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)

#visualize
  cv2.imshow('result', result_img)
  if cv2.waitKey(1) == ord('q'):
    break

  out.write(result_img)

capture.release()
out.release()
cv2.destroyAllWindows()
