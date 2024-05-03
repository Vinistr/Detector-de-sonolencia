# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import subprocess as sp
import json
import ffmpeg
import os


def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotate_cv = None
    rotate_ff = None
    """if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
         rotateCode = cv2.ROTATE_90_CLOCKWISE
     elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
         rotateCode = cv2.ROTATE_180
     elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
         rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE"""

    if 'streams' in meta_dict:
        for stream in meta_dict['streams'][0]['tags']:
            if stream == 'rotate':
                # print your info here (I'm abbreviating)
                if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
                    rotate_cv = cv2.ROTATE_90_CLOCKWISE
                    rotate_ff = 'transpose=2'
                elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
                    rotate_cv = cv2.ROTATE_180
                    rotate_ff = 'transpose=2,transpose=2'
                elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
                    rotate_cv = cv2.ROTATE_90_COUNTERCLOCKWISE
                    rotate_ff = 'transpose=2'

    return rotate_cv, rotate_ff


def correct_rotation(frame, rotate_code):
    return cv2.rotate(frame, rotate_code)


# video = "C:/Users/vinicius.sartore/Documents/TCC/drowsiness-detection/drowsiness-detection/video teste tcc3.mp4"
video = "C:/Users/vinicius.sartore/Documents/TCC/drowsiness-detection/drowsiness-detection/teste3.mp4"
#video = ':0'

archive = os.path.basename(video)

# start the file video stream thread and allow the buffer to
# start to fill
print("[INFO] starting video file thread...")
# fvs = FileVideoStream(video).start()
time.sleep(1.0)
# start the FPS timer
fps = FPS().start()
rotate_code = None
# Read video width, height and framerate using OpenCV (use it if you don't know the size of the video frames).
#rotate_code, transpose = check_rotation(video)
cap = cv2.VideoCapture(video)
framerate = cap.get(5)  # frame rate
# Get resolution of input video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Release VideoCapture - it was used just for getting video resolution
cap.release()

if rotate_code is None:
    command = ['ffmpeg.exe',
               '-i', video,
               '-f', 'image2pipe',  # Use image2pipe demuxer
               '-pix_fmt', 'bgr24',  # Set BGR pixel format
               '-vcodec', 'rawvideo',  # Get rawvideo output format.
               '-']
else:
    command = ['ffmpeg.exe',
               '-i', video,
               '-f', 'image2pipe',  # Use image2pipe demuxer
               '-pix_fmt', 'bgr24',  # Set BGR pixel format
               '-vcodec', 'rawvideo',  # Get rawvideo output format.
               '-vf', transpose,'-']

"""command = ['ffmpeg.exe',
               '-i', video,
               '-f', 'image2pipe',  # Use image2pipe demuxer
               '-pix_fmt', 'bgr24',  # Set BGR pixel format
               '-vcodec', 'rawvideo',  # Get rawvideo output format.
               '-vf', 'transpose=1','-']"""

proc = sp.Popen(command, stdout=sp.PIPE)

# loop over frames from the video file stream
# while fvs.more():
while True:
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale (while still retaining 3
    # channels)
    raw_frame = proc.stdout.read(width * height * 3)

    if len(raw_frame) != (width * height * 3):
        print('Error reading frame!!!')  # Break the loop in case of an error (too few bytes were read).
        break

    # frame = fvs.read()

    if raw_frame is None:
        break

    frame = np.fromstring(raw_frame, np.uint8)
    frame = frame.reshape((height, width, 3))
    proc.stdout.flush()
    #frame = imutils.resize(frame, width=650)
    # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotate_code is not None:
        frame = correct_rotation(frame, rotate_code)

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = np.dstack([frame, frame, frame])
    # display the size of the queue on the frame
    # cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
    #    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # show the frame and update the FPS counter
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Se a tecla 'q' for pressionada, sai do loop
    if key == ord("q"):
        break

    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
# fvs.stop()
