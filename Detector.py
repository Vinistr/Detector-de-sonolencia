from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils.video import FileVideoStream
from imutils.video import FPS
from imutils import face_utils
from threading import Thread
import playsound
import imutils
import time
import dlib
import cv2
import ffmpeg
import os
import datetime
import PySimpleGUI as sg
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt

layout = [[sg.Text('Executando...')],
          [sg.ProgressBar(100, orientation='h', size=(20, 20), key='progressbar')],
          [sg.Cancel()]]

window = sg.Window('Video em análise', layout)
progress_bar = window['progressbar']

# Constantes
EYE_AR_THRESH = 0.18  # .03
EYE_AR_CONSEC_FRAMES = 8  # 48
MOU_AR_THRESH = 0.35  # 0.75
MOU_AR_CONSEC_FRAMES = 12


def monta_grafico(arquivo):
    SHOWCASE_DATA = pd.read_csv(arquivo)
    SHOWCASE_DATA_CUMSUM = SHOWCASE_DATA.cumsum(axis=0)
    x_total = SHOWCASE_DATA_CUMSUM.frame.size
    total_sleeps = SHOWCASE_DATA['sleeps'].sum()
    total_yawns = SHOWCASE_DATA['yawns'].sum()
    SHOWCASE_DATA_CUMSUM = SHOWCASE_DATA_CUMSUM.drop('ear', 1)
    DF_SLEEP = SHOWCASE_DATA
    DF_YAWN = SHOWCASE_DATA
    DF_SLEEP = DF_SLEEP[DF_SLEEP.sleeps > 0]
    DF_YAWN = DF_YAWN[DF_YAWN.yawns > 0]

    plot = SHOWCASE_DATA.ear.plot(color="blue", label="EAR", linewidth=1.0)
    plot = SHOWCASE_DATA.mar.plot(color="purple", label="MAR", linewidth=1.0)
    plt.plot(DF_SLEEP.index, DF_SLEEP.ear, 'o', color="red", label="Sonolência")
    plt.plot(DF_YAWN.index, DF_YAWN.mar, 'o', color="cyan", label="Bocejo")
    plt.ylabel("Aspect Ratio")
    plt.xlabel("Frames")
    plt.title("Monitoramento - " + str(os.path.splitext(os.path.basename(arquivo))[0])) #nome do arquivo
    plt.hlines(y=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], xmin=0, xmax=x_total, color='black',
               linestyle='--')
    plt.legend()

    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    plt.annotate("Total de sonolências: " + str(total_sleeps) + "\n" + "Total de bocejos: " + str(total_yawns),
                 xy=(0.05, 0.99), xycoords='axes fraction', fontsize=16,
                 horizontalalignment='left', verticalalignment='top', bbox=props)

    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    plt.show()


def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotate_cv = None
    rotate_ff = None

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


def sound_alarm():
    # play an alarm sound
    playsound.playsound("alarm.wav")


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    c = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (a + b) / (2.0 * c)

    # return the eye aspect ratio
    return ear


def mouth_aspect_ratio(mouth):
    # Inner mouth
    # compute the euclidean distances between the vertical
    a = dist.euclidean(mouth[1], mouth[7])
    b = dist.euclidean(mouth[3], mouth[5])
    # compute the euclidean distances between the horizontal
    c = dist.euclidean(mouth[0], mouth[4])

    # compute mouth aspect ratio
    mor = (a + b) / (2.0 * c)
    return mor


def detector(entrada, video, alerta, exibe_video, arq_webcam):
    # inicializa o contador de frames, bem como um booleano usado para indicar se o alarme está disparando
    counter = 0
    alarm_on = False
    sleep_status = False
    sleeps = 0

    counter_yawn = 0
    yawn_status = False
    yawns = 0

    # inicializa o detector facial (HOG-based) e o modelo de mapeamento de pontos de interesses faciais
    print("[INFO] Carregando preditor de pontos de interesses faciais ...")
    detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    predictor = dlib.shape_predictor("eye_mouth_predictor.dat")

    FACIAL_LANDMARKS_IDXS = OrderedDict([
        ("inner_mouth", (12, 20)),
        ("right_eye", (0, 6)),
        ("left_eye", (6, 12)),
    ])

    # índices dos pontos faciais, olhos e boca
    (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = FACIAL_LANDMARKS_IDXS["inner_mouth"]

    # inicia a transmissão do vídeo ou webcam
    print("[INFO] Iniciando a transmissão do vídeo...")

    # o FileVideoStream fica lento pra reproduzir a webcam mas funciona com videos e o VideoStream funciona pra webcam mas fica lento em videos
    if entrada == 1:

        # Read video width, height and framerate using OpenCV (use it if you don't know the size of the video frames).
        # check_rotation verifica a orientação do vídeo, se estiver na vertical, tem que tratar
        # porque o opencv não trata isso e deixa o vídeo sempre na horizontal
        rotate_code, transpose = check_rotation(video)
        cap = cv2.VideoCapture(video)
        # Get resolution of input video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Release VideoCapture - it was used just for getting video resolution

        # count the number of frames
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps_tot = int(cap.get(cv2.CAP_PROP_FPS))
        # calculate dusration of the video
        seconds = int(frames / fps_tot)
        video_time = str(datetime.timedelta(seconds=seconds))
        cap.release()

        arquivo_monitoramento = "videos/" + os.path.basename(video) + "_monitoramento.csv"
        vs = FileVideoStream(video).start()
    else:
        arquivo_monitoramento = "videos/webcam_" + arq_webcam + ".csv"
        if os.path.isfile(arquivo_monitoramento):
            base = os.path.basename(arquivo_monitoramento)
            nome = os.path.splitext(base)[0]
            i = int(nome[len(nome) - 1]) + 1
            arquivo_monitoramento = "videos/webcam_" + arq_webcam + str(i) + ".csv"
        vs = VideoStream(0).start()

    time.sleep(1.0)
    fps = FPS().start()
    frame_number = 0

    # cria o arquivo de monitoramento
    with open(arquivo_monitoramento, "w") as arquivo:
        arquivo.write("frame,ear,sleep_status,sleeps,mar,yawn_status,yawns" + "\n")

    # loop durante a execução do vídeo
    i = 0
    while True:

        if exibe_video == 0:
            event, values = window.read(timeout=10)
            # check to see if the cancel button was clicked and exit loop if clicked
            if event == 'Cancel' or event == sg.WIN_CLOSED:
                break

        # pega cada frame do video, faz o redimensionamento e converte em cinza
        # essa conversão em cinza ajuda a acelerar o processo de detecção facial e mapeamento dos pontos
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale (while still retaining 3
        # channels)
        frame = vs.read()
        if frame is None:
            break

        if entrada == 1:
            # redimensiona o tamanho do video
            # para melhorar o desempenho em arquivos de video, na webcam não precisa
            frame = imutils.resize(frame, width=400)
            # check if the frame needs to be rotated
            if rotate_code is not None:
                frame = correct_rotation(frame, rotate_code)

        # Pega o tamanho da tela para posicionar os labels
        window_width = frame.shape[1]
        window_height = frame.shape[0]

        # conver o frame da imagem em preto e branco
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # variáveis que controlam o status da sonolência e bocejo
        prev_yawn_status = yawn_status
        prev_yawns = yawns
        prev_sleep_status = sleep_status
        prev_sleeps = sleeps

        # detecta uma face no frame
        rects = detector(gray, 0)

        # faz um loop nas faces detectadas
        for rect in rects:
            # faz o mapeamento dos pontos de interasse na região da face que foi detectada
            # converte esses pontos (coordenadas x,y) em um array NumPy
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extrais os pontos dos olhos e boca, e os pontos são utilizados
            # para calcular respectivamento o EAR e o MAR
            left_eye = shape[lStart:lEnd]
            right_eye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            mouth_mar = mouth_aspect_ratio(mouth)

            # Calcula a média do EAR para ambos os olhos
            ear = (left_ear + right_ear) / 2.0

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            if exibe_video == 1:
                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                mouth_hull = cv2.convexHull(mouth)
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)

            # EAR
            # verifica se o ear é menor que a constante, se sim incrementa o contador
            if ear < EYE_AR_THRESH and yawn_status == False:
                counter += 1

                # se o contador for incrementando um número de vezes que supera a constante
                # então significa que a pessoa está possívelmente dormindo,
                if counter >= EYE_AR_CONSEC_FRAMES:
                    # ativa o alarme
                    sleep_status = True
                    if exibe_video == 1:
                        if not alarm_on and alerta == 1:
                            alarm_on = True

                            # O alarme é executado por uma Thread, com isso ele pode ficar sendo executando em segundo plano
                            t = Thread(target=sound_alarm)
                            t.deamon = True
                            t.start()

                        # Escreve o aviso do alarme na tela
                        cv2.putText(frame, "ALERTA DE SONOLENCIA!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 255), 2)

                else:
                    sleep_status = False

            # se não for, reseta o contador e o alarme
            else:
                counter = 0
                alarm_on = False
                if prev_sleep_status == True and sleep_status == True:
                    sleep_status = False

            # Mostra o valor do ear na imagem, como uma forma de acompanhar o valor durante a execução do vídeo
            # o sleeps é um contador com as vezes que o usuário dormiu
            if prev_sleep_status == True and sleep_status == False:
                sleeps += 1

            cv2.putText(frame, "EAR: {:.2f}".format(ear), (window_width - window_width + 10, window_height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # MAR
            if mouth_mar > MOU_AR_THRESH:
                counter_yawn += 1

                if counter_yawn >= MOU_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "BOCEJO, ALERTA DE SONOLENCIA! ", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 2)
                    yawn_status = True
                else:
                    yawn_status = False
            else:
                counter_yawn = 0
                if prev_yawn_status == True and yawn_status == True:
                    yawn_status = False

            if prev_yawn_status == True and yawn_status == False:
                yawns += 1

            # Mesma ideia de acompanhar o valor do ear feita anteriormente, só que agora para o mar
            cv2.putText(frame, "MAR: {:.2f}".format(mouth_mar), (window_width - window_width + 10, window_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # controle para detectar o fim de uma sonolência ou bocejo
        # funcionara como um marcador no arquivo
        if sleeps > prev_sleeps:
            sleeps_ctrl = 1
        else:
            sleeps_ctrl = 0

        if yawns > prev_yawns:
            yawns_ctrl = 1
        else:
            yawns_ctrl = 0

        # if frame_number > fps_tot:
        with open(arquivo_monitoramento, "a") as arquivo:
            arquivo.write(str(frame_number) + "," +
                          str("{:.2f}".format(ear)) + "," +
                          str(sleep_status) + "," + str(sleeps_ctrl) + "," +
                          str("{:.2f}".format(mouth_mar)) + "," +
                          str(yawn_status) + "," + str(yawns_ctrl) +
                          "\n")

        fps.update()
        frame_number += 1
        # elapsed = frame_number / fps_tot

        if exibe_video == 0:
            i += 1
            progress_bar.UpdateBar(i + 1, frames)

        '''frame_number += 1
        #elapsed = datetime.datetime.now()
        elapsed = frame_number / fps_tot
        time.sleep(0.01)
        print("[INFO] elasped time during video: " + str(elapsed))
        cv2.putText(frame, "Tempo: {:.2f}".format(elapsed), (window_width - 150, window_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)'''

        # Mostra a janela com as imagens do video
        if exibe_video == 1:
            cv2.imshow("Frame", frame)
            cv2.moveWindow("Frame", 40, 30)

        key = cv2.waitKey(1) & 0xFF

        # Se a tecla 'q' for pressionada, sai do loop
        if key == ord("q"):
            break

    window.close()
    # stop the timer and display FPS information
    fps.stop()

    print("[INFO] final elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # Fecha a janela e para a execução
    cv2.destroyAllWindows()
    vs.stop()
