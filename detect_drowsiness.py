# USAGE
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils.video import FileVideoStream
from imutils.video import FPS
from imutils import face_utils
from threading import Thread
from multiprocessing.pool import ThreadPool
from queue import Queue
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

def sound_alarm():
    # play an alarm sound
    playsound.playsound("alarm.wav")

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

def MOR(mouth):
    #All mouth
    # compute the euclidean distances between the horizontal
    #point = dist.euclidean(mouth[0], mouth[6])
    # compute the euclidean distances between the vertical
    #point1 = dist.euclidean(mouth[2], mouth[10])
    #point2 = dist.euclidean(mouth[4], mouth[8])
    # taking average

    #Inner mouth
    A = dist.euclidean(mouth[1], mouth[7])
    B = dist.euclidean(mouth[3], mouth[5])
    C = dist.euclidean(mouth[0], mouth[4])

    # compute mouth aspect ratio
    mor = (A + B) / (2.0 * C)
    return mor

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--shape-predictor", required=True,
#	help="path to facial landmark predictor")
#ap.add_argument("-a", "--alarm", type=str, default="",
#	help="path alarm .WAV file")
#ap.add_argument("-w", "--webcam", type=int, default=0,
#	help="index of webcam on system")
args = vars(ap.parse_args())
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.18  #.03
EYE_AR_CONSEC_FRAMES = 8 #48
MOU_AR_THRESH = 0.35 #0.75

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
yawnStatus = False
yawns = 0
ALARM_ON = False
#output_text = ""

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] carregando preditor de pontos de interesse ...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

# start the video stream thread
print("[INFO] Iniciando transmissão do vídeo...")

# o FileVideoStream fica lento pra reproduzir a webcam mas funciona com videos e o VideoStream funciona pra webcam mas fica lento em videos
#vs = VideoStream(src=args["webcam"]).start()
#vs = VideoStream(0).start()
#vs = VideoStream("C:/Users/vinicius.sartore/Documents/TCC/drowsiness-detection/drowsiness-detection/video teste tcc3.mp4").start()

vs = FileVideoStream("C:/Users/vinicius.sartore/Documents/TCC/drowsiness-detection/drowsiness-detection/video teste tcc3.mp4").start()
#vs = FileVideoStream(0).start()


time.sleep(1.0)
fps = FPS().start()

# loop over frames from the video stream
while True:
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    twrv = ThreadWithReturnValue(target=vs.read,args=())
    twrv.start()
    frame = twrv.join()
    #frame = vs.read()

    if frame is None:
        break

    twrv = ThreadWithReturnValue(target=imutils.resize,args=(frame, 650))
    twrv.start()
    frame = twrv.join()
    #frame = imutils.resize(frame, width=650)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = np.dstack([gray, gray, gray])
    prev_yawn_status = yawnStatus

    #cv2.putText(frame, "Queue Size: {}".format(vs.Q.qsize()),
    #    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    '''twrv = ThreadWithReturnValue(target=detector,args=(frame, 0))
    twrv.start()
    rects = twrv.join()   # prints foo'''
    rects = [(185, 90), (400, 305)]

    """pool = ThreadPool(processes=8)
    async_result = pool.apply_async(detector, (frame, 0)) # tuple of args for foo
    # do some other stuff in the main process
    rects = async_result.get()  # get the return value from your function."""

    # detect faces in the grayscale frame
    #rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        #shape = predictor(gray, rect)
        #shape = face_utils.shape_to_np(shape)
        twrv = ThreadWithReturnValue(target=predictor,args=(gray, rect))
        twrv.start()
        shape = twrv.join()
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        '''leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        mouEAR = MOR(mouth)'''
        twrv = ThreadWithReturnValue(target=eye_aspect_ratio,args=(leftEye,))
        twrv.start()
        leftEAR = twrv.join()
        twrv = ThreadWithReturnValue(target=eye_aspect_ratio,args=(rightEye,))
        twrv.start()
        rightEAR = twrv.join()
        twrv = ThreadWithReturnValue(target=MOR,args=(mouth,))
        twrv.start()
        mouEAR = twrv.join()

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH and yawnStatus == False:
            COUNTER += 1

            # if the eyes were closed for a sufficient number of
            # then sound the alarm
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # if the alarm is not on, turn it on
                if not ALARM_ON:
                    ALARM_ON = True

                    # check to see if an alarm file was supplied,
                    # and if so, start a thread to have the alarm
                    # sound played in the background
                    t = Thread(target=sound_alarm)
                    t.deamon = True
                    t.start()

                # draw an alarm on the frame
                cv2.putText(frame, "ALERTA DE SONOLENCIA!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # otherwise, the eye aspect ratio is not below the blink
        # threshold, so reset the counter and alarm
        else:
            COUNTER = 0
            ALARM_ON = False

        # draw the computed eye aspect ratio on the frame to help
        # with debugging and setting the correct eye aspect ratio
        # thresholds and frame counters
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (480, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if mouEAR > MOU_AR_THRESH:
            cv2.putText(frame, "BOCEJO, ALERTA DE SONOLENCIA! ", (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            yawnStatus = True
            #output_text = "Bocejo Contador: " + str(yawns)
            #cv2.putText(frame, output_text, (10,300),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0),2)
        else:
            yawnStatus = False

        if prev_yawn_status == True and yawnStatus == False:
            yawns+=1

        output_text = "Bocejo Contador: " + str(yawns)
        cv2.putText(frame, output_text, (10,300),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0),2)
        cv2.putText(frame, "MAR: {:.2f}".format(mouEAR), (480, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #cv2.putText(frame, output_text, (10,300),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0),2)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    fps.update()
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
vs.stop()
