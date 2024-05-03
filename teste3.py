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
EYE_AR_THRESH = 0.18
EYE_AR_CONSEC_FRAMES = 8
MOU_AR_THRESH = 0.35
MOU_AR_CONSEC_FRAMES = 12


def monta_grafico(arquivo):
    #faz a leitura do arquvio CSV
    SHOWCASE_DATA = pd.read_csv(arquivo)
    SHOWCASE_DATA_CUMSUM = SHOWCASE_DATA.cumsum(axis=0)
    x_total = SHOWCASE_DATA_CUMSUM.frame.size #pega o valor total para desenhar o limite da linha tracejada
    total_sleeps = SHOWCASE_DATA['sleeps_ctrl'].sum() #soma da coluna para mostrar o total na legenda
    total_yawns = SHOWCASE_DATA['yawns_ctrl'].sum() #soma da coluna para mostrar o total na legenda
    SHOWCASE_DATA_CUMSUM = SHOWCASE_DATA_CUMSUM.drop('ear', 1)
    DF_SLEEP = SHOWCASE_DATA
    DF_YAWN = SHOWCASE_DATA
    DF_SLEEP = DF_SLEEP[DF_SLEEP.sleeps > 0] #pontos onde houveram mudança de status
    DF_YAWN = DF_YAWN[DF_YAWN.yawns > 0] #pontos onde houveram mudança de status
    #plota os valores do ear e mar, e os pontos que ocorreram mudanças de status
    plot = SHOWCASE_DATA.ear.plot(color="blue", label="EAR", linewidth=1.0)
    plot = SHOWCASE_DATA.mar.plot(color="purple", label="MAR", linewidth=1.0)
    plt.plot(DF_SLEEP.index, DF_SLEEP.ear, 'o', color="red", label="Sonolência")
    plt.plot(DF_YAWN.index, DF_YAWN.mar, 'o', color="cyan", label="Bocejo")
    plt.ylabel("Aspect Ratio")
    plt.xlabel("Frames")
    plt.title("Monitoramento")
    plt.hlines(y=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], xmin=0, xmax=x_total, color='black', linestyle='--')
    plt.legend()
    #legenda dos totais de sonolencia e bocejo
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    plt.annotate("Total de sonolências: " + str(total_sleeps) + "\n" + "Total de bocejos: " + str(total_yawns),
                 xy=(0.05, 0.99), xycoords='axes fraction', fontsize=16,
                 horizontalalignment='left', verticalalignment='top', bbox=props)
    #exibição do gráfico em tela cheia
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    plt.show()


def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)
    rotate_cv = None

    if 'streams' in meta_dict:
        for stream in meta_dict['streams'][0]['tags']:
            if stream == 'rotate':
                if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
                    rotate_cv = cv2.ROTATE_90_CLOCKWISE
                elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
                    rotate_cv = cv2.ROTATE_180
                elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
                    rotate_cv = cv2.ROTATE_90_COUNTERCLOCKWISE

    return rotate_cv


def correct_rotation(frame, rotate_code):
    return cv2.rotate(frame, rotate_code)


def sound_alarm():
    # toca o seguinte aúdio
    playsound.playsound("alarm.wav")


def eye_aspect_ratio(eye):
    # obtem a distancia entre os pontos verticais
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])

    # obtem a distancia entre os pontos horizontais
    c = dist.euclidean(eye[0], eye[3])

    # calculo do ear
    ear = (a + b) / (2.0 * c)

    return ear


def mouth_aspect_ratio(mouth):
    # obtem a distancia entre os pontos verticais
    a = dist.euclidean(mouth[1], mouth[7])
    b = dist.euclidean(mouth[3], mouth[5])

    # obtem a distancia entre os pontos horizontais
    c = dist.euclidean(mouth[0], mouth[4])

    # calculo do mar
    mar = (a + b) / (2.0 * c)

    return mar


def detector(entrada, video, alerta, exibe_video, arq_webcam):
    print("[INFO] Carregando preditor de pontos de interesses faciais ...")
    # objeto de detecção facial
    detector = dlib.get_frontal_face_detector()
    # objeto de mapeamento de pontos de interesse
    predictor = dlib.shape_predictor("eye_mouth_predictor.dat")

    # Dicionário dos pontos de interesse
    FACIAL_LANDMARKS_IDXS = OrderedDict([
        ("inner_mouth", (12, 20)),
        ("right_eye", (0, 6)),
        ("left_eye", (6, 12)),
    ])

    # marcação do ponto inicial e final de cada parte a ser mapeada
    (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = FACIAL_LANDMARKS_IDXS["inner_mouth"]

    print("[INFO] Iniciando a transmissão do vídeo...")
    if entrada == 1:

        rotate_code = check_rotation(video)

        cap = cv2.VideoCapture(video)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        vs = FileVideoStream(video).start()
    else:
        vs = VideoStream(0).start()

    # inicialização do arquivo
    if entrada == 1:
        arquivo_monitoramento = "videos/" + os.path.basename(video) + "_monitoramento.csv"
    else:
        arquivo_monitoramento = "videos/webcam_" + arq_webcam + ".csv"
        if os.path.isfile(arquivo_monitoramento):
            base = os.path.basename(arquivo_monitoramento)
            nome = os.path.splitext(base)[0]
            i = int(nome[len(nome) - 1]) + 1
            arquivo_monitoramento = "videos/webcam_" + arq_webcam + str(i) + ".csv"

    with open(arquivo_monitoramento, "w") as arquivo:
        arquivo.write("frame,ear,sleep_status,sleeps_ctrl,mar,yawn_status,yawns_ctrl" + "\n")

    alarm_on = False
    # contador de frames de sonolencia, estado de sonolencia, contador de sonolencia
    counter_sleep = 0
    sleep_status = False
    sleeps = 0
    # contador de frames de sonolencia, estado de sonolencia, contador de sonolencia
    counter_yawn = 0
    yawn_status = False
    yawns = 0

    time.sleep(1.0)
    fps = FPS().start()
    frame_number = 0
    i = 0

    # Loop principal
    while True:

        # exibe barra de progresso quando for escolhida a opção "gerar apenas arquivo"
        if exibe_video == 0:
            event, values = window.read(timeout=10)
            if event == 'Cancel' or event == sg.WIN_CLOSED:
                break
        # le um frame do video
        frame = vs.read()
        if frame is None:
            break

        if entrada == 1:
            # redimensionamento
            frame = imutils.resize(frame, width=450)
            # verifica se a tela precisa ser rotacionada
            if rotate_code is not None:
                frame = correct_rotation(frame, rotate_code)

        # converte o frame em escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Pega o tamanho da tela para posicionar os labels de alerta
        window_width = frame.shape[1]
        window_height = frame.shape[0]

        # variáveis que controlam o status anterior da sonolência e bocejo
        prev_yawn_status = yawn_status
        prev_yawns = yawns
        prev_sleep_status = sleep_status
        prev_sleeps = sleeps

        # detecção facial
        rects = detector(gray, 0)

        for rect in rects:
            # mapeamento dos pontos
            shape = predictor(gray, rect)
            # faz a conversão para um array de duas dimensões
            shape = face_utils.shape_to_np(shape)

            # armazena nas variaveis os pontos encontrados
            left_eye = shape[lStart:lEnd]
            right_eye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]

            # aqui as variaveis são usadas para desenhar na tela
            # os pontos ligados por linhas
            if exibe_video == 1:
                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                mouth_hull = cv2.convexHull(mouth)
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            mouth_mar = mouth_aspect_ratio(mouth)

            ear = (left_ear + right_ear) / 2.0

            cv2.putText(frame, "EAR: {:.2f}".format(ear), (window_width - window_width + 10, window_height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.putText(frame, "MAR: {:.2f}".format(mouth_mar), (window_width - window_width + 10, window_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # EAR
            if ear < EYE_AR_THRESH and yawn_status == False:
                counter_sleep += 1
                #se o EAR for menor que a constante e o contador de frames for maior que a constante de frames
                #enão entrou em sonolência
                if counter_sleep >= EYE_AR_CONSEC_FRAMES:
                    sleep_status = True
                    # quando não exibir video, não faz sentido tocar alarme e nem imprimir valores na tela
                    if exibe_video == 1:
                        #dispara o alarme, se foi definido para tocar
                        if not alarm_on and alerta == 1:
                            alarm_on = True

                            t = Thread(target=sound_alarm)
                            t.deamon = True
                            t.start()

                        cv2.putText(frame, "ALERTA DE SONOLENCIA!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 255), 2)

                else:
                    sleep_status = False

            else:
                counter_sleep = 0
                alarm_on = False
                if prev_sleep_status == True and sleep_status == True:
                    sleep_status = False

            if prev_sleep_status == True and sleep_status == False:
                sleeps += 1

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

        #Flags que indicam quando ocorreu uma sonolência ou bocejo
        #Valores sempre serão 0, e quando o contador for maior que
        #o valor do contador anterior, significa mudança de status
        #esse flag será usado para marcar no gráfico os pontos nos quais ocorreram a mudança de status
        if sleeps > prev_sleeps:
            sleeps_ctrl = 1
        else:
            sleeps_ctrl = 0

        if yawns > prev_yawns:
            yawns_ctrl = 1
        else:
            yawns_ctrl = 0

        #Inclui uma linha no arquivo
        with open(arquivo_monitoramento, "a") as arquivo:
            arquivo.write(str(frame_number) + "," +
                          str("{:.2f}".format(ear)) + "," +
                          str(sleep_status) + "," + str(sleeps_ctrl) + "," +
                          str("{:.2f}".format(mouth_mar)) + "," +
                          str(yawn_status) + "," + str(yawns_ctrl) +
                          "\n")

        fps.update()
        frame_number += 1

        if exibe_video == 0:
            i += 1
            progress_bar.UpdateBar(i + 1, frames)

        if exibe_video == 1:
            cv2.imshow("Frame", frame)
            cv2.moveWindow("Frame", 40, 30)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    window.close()
    fps.stop()

    print("[INFO] final elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # Fecha a janela e para a execução
    cv2.destroyAllWindows()
    vs.stop()
