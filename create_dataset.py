from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils.video import FileVideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2

def MOR(drivermouth):
    # compute the euclidean distances between the horizontal
    point = dist.euclidean(drivermouth[0], drivermouth[6])
    # compute the euclidean distances between the vertical
    point1 = dist.euclidean(drivermouth[2], drivermouth[10])
    point2 = dist.euclidean(drivermouth[4], drivermouth[8])
    # taking average
    ypoint = (point1+point2)/2.0
    # compute mouth aspect ratio
    mouth_aspect_ratio = ypoint/point
    return mouth_aspect_ratio

MOU_AR_THRESH = 0.75 #0.75

counter = 0
yawnStatus = False
yawns = 0

# We will save images after every 4 frames
# This is done so we don't have lot's of duplicate images
skip_frames = 3
frame_gap = 0
directory = 'train_images_h'
box_file = 'boxes_h.txt'

print("[INFO] carregando preditor de marco facial ...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
#(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

print("[INFO] Iniciando transmissão do vídeo...")
#vs = FileVideoStream("videos/Male/1-MaleGlasses.avi").start()
vs = FileVideoStream("videos/Male/1-MaleGlasses.avi").start()

fr = open(box_file, 'a')

time.sleep(1.0)


# loop over frames from the video stream
while True:
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    frame = vs.read()
    frame = imutils.resize(frame, width=650)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_yawn_status = yawnStatus

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        orig = frame.copy()

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        #leftEye = shape[lStart:lEnd]
        #rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        #leftEAR = eye_aspect_ratio(leftEye)
        #rightEAR = eye_aspect_ratio(rightEye)
        mouEAR = MOR(mouth)

        # average the eye aspect ratio together for both eyes
        #ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        #leftEyeHull = cv2.convexHull(leftEye)
        #rightEyeHull = cv2.convexHull(rightEye)
        #mouthHull = cv2.convexHull(mouth)
        #cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        #cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        #cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        if mouEAR > MOU_AR_THRESH:

            yawnStatus = True
            frame_gap +=1
            if frame_gap == skip_frames:
                e, t, d, b = (int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom()) )
                #roi = frame[t:b, e:d].copy()
                # Set the image name equal to the counter value
                img_name = str(counter)  + '.png'

                # Save the Image in the defined directory
                img_full_name = directory + '/' + str(counter) +  '.png'
                cv2.imwrite(img_full_name, orig)

                # Save the bounding box coordinates in the text file.
                fr.write('{}:({},{},{},{}),'.format(counter, e, t, d, b))

                counter += 1
                frame_gap = 0

        else:
            yawnStatus = False
        if prev_yawn_status == True and yawnStatus == False:
            yawns+=1
        cv2.putText(frame, "MAR: {:.2f}".format(mouEAR), (480, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
fr.close()
