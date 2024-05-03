import PySimpleGUI as sg
from Detector import detector, monta_grafico
import os


class TelaPython:
    def __init__(self):
        # Layout
        layout = [
            [sg.Text("Selecione o caminho do vídeo")],
            [sg.Text("Vídeo", size=(5, 1)), sg.InputText(size=(100, 1), key="caminho"),
             sg.FileBrowse("Arquivo", key="browse", file_types=(("Video Files", "*.mp4"), ("Video Files", "*.mov")))],
            [sg.Checkbox("Ligar alerta sonoro durante o monitoramento", key="alerta")],
            [sg.Text(" ", size=(41, 1))],
            [sg.Text("Selecione a forma de monitoramento")],
            [sg.Button("Webcam", key="webcam", button_color=("white", "blue"), size=(6, 1)),
             sg.Button("Video", key="video", button_color=("white", "green"), size=(6, 1)),
             sg.Button("Gerar apenas arquivos (Sem exibir video)", key="sem_exibir_video",
                       button_color=("white", "black"), size=(30, 1)),
             sg.Text(" ", size=(55, 1))],
            [sg.Text(" ", size=(41, 1))],
            [sg.Text("Selecione um arquivo de monitoramento para visualizar seu gráfico")],
            [sg.Text("Arquivo", size=(5, 1)), sg.InputText(size=(100, 1), key="caminho_grafico"),
             sg.FileBrowse("Arquivo", key="browse", file_types=(("CSV Files", "*.csv"),)),
             sg.Button("Gerar gráfico", key="grafico", button_color=("black", "yellow"), size=(12, 1))],
            [sg.Text(" ", size=(41, 1))],
            [sg.Text(" ", size=(41, 1))],
            [sg.Exit("Sair", button_color=("white", "red"), size=(6, 1))]
        ]

        # Janela
        self.janela = sg.Window("Monitoramento de sonolência").layout(layout)
        # Extrair os dados da tela
        # self.button, self.values = self.janela.Read()

    def Iniciar(self):
        while True:
            # Extrair os dados da tela
            self.button, self.values = self.janela.Read()
            print(self.button, self.values)
            if self.button in (None, "Sair"):
                break

            if self.button == "webcam":
                detector(0, "", self.values["alerta"], 1)

            if self.button == "video":
                if self.values["caminho"] == '':
                    sg.Popup('O campo "vídeo" está vazio!', title='Erro')
                elif not os.path.isfile(self.values["caminho"]):
                    sg.Popup('O caminho informado no campo "vídeo" não é valido!', title='Erro')
                else:
                    detector(1, self.values["caminho"], self.values["alerta"], 1)

            if self.button == "sem_exibir_video":
                if self.values["caminho"] == '':
                    sg.Popup('O campo "vídeo" está vazio!', title='Erro')
                elif not os.path.isfile(self.values["caminho"]):
                    sg.Popup('O caminho informado no campo "vídeo" não é valido!', title='Erro')
                else:
                    detector(1, self.values["caminho"], 0, 0)

            if self.button == "grafico":
                if self.values["caminho_grafico"] == '':
                    sg.Popup('O campo "gráfico" está vazio!', title='Erro')
                elif not os.path.isfile(self.values["caminho_grafico"]):
                    sg.Popup('O caminho informado no campo "gráfico"  não é valido!', title='Erro')
                else:
                    monta_grafico(self.values["caminho_grafico"])


tela = TelaPython()
tela.Iniciar()
