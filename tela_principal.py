import PySimpleGUI as sg
from Detector import detector, monta_grafico
import os


def janela_principal():
    # Layout
    layout = [
        [sg.Text("Selecione a forma de monitoramento")],
        [sg.Button("Webcam", key="webcam", button_color=("white", "blue"), size=(6, 1)),
         sg.Button("Video", key="video", button_color=("white", "green"), size=(6, 1)),
         sg.Button("Gerar apenas arquivo (Sem exibir video)", key="sem_exibir_video",
                   button_color=("white", "black"), size=(30, 1)),
         sg.Button("Gerar gráfico", key="grafico", button_color=("black", "yellow"), size=(12, 1))],
        [sg.Text(" ", size=(41, 1))],
        [sg.Text(" ", size=(41, 1))],
        [sg.Exit("Sair", key="sair", button_color=("white", "red"), size=(6, 1))]
    ]
    return sg.Window('Menu Principal', layout=layout, finalize=True)

def janela_nome_arq():
    # Layout
    layout = [
        [sg.Text("Informe o nome do arquivo de monitoramento")],
        [sg.Text("Arquivo", size=(5, 1)), sg.InputText(size=(100, 1), key="nome_arq")],
        [sg.Checkbox("Ligar alerta sonoro durante o monitoramento", key="alerta")],
        [sg.Button("Ok", key="ok"),
         sg.Button("Voltar", key="voltar")]
    ]

    return sg.Window('Nome arquivo', layout=layout, finalize=True)

def janela_pesq_video():
    # Layout
    layout = [
        [sg.Text("Selecione o caminho do vídeo")],
        [sg.Text("Vídeo", size=(5, 1)), sg.InputText(size=(100, 1), key="caminho"),
         sg.FileBrowse("Arquivo", key="browse", file_types=(("Video Files", "*.mp4"), ("Video Files", "*.mov")))],
        [sg.Checkbox("Ligar alerta sonoro durante o monitoramento", key="alerta")],
        [sg.Button("Ok", key="ok"),
         sg.Button("Voltar", key="voltar")]
    ]

    return sg.Window('Pesquisa video', layout=layout, finalize=True)


def janela_pesq_grafico():
    # Layout
    layout = [
        [sg.Text("Selecione um arquivo de monitoramento para visualizar seu gráfico")],
        [sg.Text("Arquivo", size=(5, 1)), sg.InputText(size=(100, 1), key="caminho_grafico"),
         sg.FileBrowse("Arquivo", key="browse", file_types=(("CSV Files", "*.csv"),))],
        [sg.Button("Ok", key="ok"),
         sg.Button("Voltar", key="voltar")]
    ]

    return sg.Window('Pesquisa video', layout=layout, finalize=True)


j_menu, j_pesq_video, j_pesq_svideo, j_pesq_graf, j_nome_arq = janela_principal(), None, None, None, None

while True:
    window, event, values = sg.read_all_windows()

    # janela principal
    if window == j_menu and event == 'sair':
        break
    if window == j_menu and event == 'webcam':
        j_nome_arq = janela_nome_arq()
        j_menu.hide()
    if window == j_menu and event == 'video':
        j_pesq_video = janela_pesq_video()
        j_menu.hide()
    if window == j_menu and event == 'sem_exibir_video':
        j_pesq_svideo = janela_pesq_video()
        j_menu.hide()
    if window == j_menu and event == 'grafico':
        j_pesq_graf = janela_pesq_grafico()
        j_menu.hide()

    #janela para digitar o nome do arquivo de monitoramento, quando for webcam
    if window == j_nome_arq and event == 'voltar':
        j_nome_arq.hide()
        j_menu.un_hide()
    if window == j_nome_arq and event == 'ok':
        if values["nome_arq"] == '':
            sg.Popup('Digite um nome para o arquivo!', title='Erro')
        else:
            detector(0, "", values["alerta"], 1, values["nome_arq"])

    #janela de pesquisa de video
    if window == j_pesq_video and event == 'voltar':
        j_pesq_video.hide()
        j_menu.un_hide()
    if window == j_pesq_video and event == 'ok':
        if values["caminho"] == '':
            sg.Popup('O campo "vídeo" está vazio!', title='Erro')
        elif not os.path.isfile(values["caminho"]):
            sg.Popup('O caminho informado no campo "vídeo" não é valido!', title='Erro')
        else:
            detector(1, values["caminho"], values["alerta"], 1, "")

    #janela de pesquisa de video, quando for a opção sem video
    if window == j_pesq_svideo and event == 'voltar':
        j_pesq_svideo.hide()
        j_menu.un_hide()
    if window == j_pesq_svideo and event == 'ok':
        if values["caminho"] == '':
            sg.Popup('O campo "vídeo" está vazio!', title='Erro')
        elif not os.path.isfile(values["caminho"]):
            sg.Popup('O caminho informado no campo "vídeo" não é valido!', title='Erro')
        else:
            detector(1, values["caminho"], 0, 0, "")

    #janela de pesquisa de gráfico
    if window == j_pesq_graf and event == 'voltar':
        j_pesq_graf.hide()
        j_menu.un_hide()
    if window == j_pesq_graf and event == 'ok':
        if values["caminho_grafico"] == '':
            sg.Popup('O campo "gráfico" está vazio!', title='Erro')
        elif not os.path.isfile(values["caminho_grafico"]):
            sg.Popup('O caminho informado no campo "gráfico"  não é valido!', title='Erro')
        else:
            monta_grafico(values["caminho_grafico"])
