import numpy as np
import cv2
import os


"""
Os vídeos invertidos estão na pasta datasets,
todos tem no final o nome _inv.avi.
out-3.ogv <-> out-3_inv.avi
"""

nome_video = "out-2.ogv"

def inverte(nome_video):

    # Pasta do vídeo original:
    dir = os.getcwd() + "/datasets/" + nome_video

    # Captura do vídeo original:
    cap = cv2.VideoCapture(dir)

    # Retorno da largura e da altura originais:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Valor da rotação, -1 significa 180 graus:
    rotacao = -1

    # Parâmetro de configuração para escrever o vídeo:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Destino do vídeo invertido e nome:
    arquivo = "datasets/" + nome_video.split(".")[0] +"_inv.avi"

    # Função para gerar o video invertido:
    out = cv2.VideoWriter(arquivo,fourcc, 20.0, (width, height))

    ret, frame = cap.read()
    # Repetição de leitura:
    while ret:

        cv2.imshow('frame',frame)

        ret, frame = cap.read()

        # Função que inverte o vídeo:
        frame = cv2.flip(frame, rotacao)

        # Função de escrita do vídeo:
        out.write(frame)

        # Saída do vídeo apertando ESC:
        c = cv2.waitKey(2)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# ========================================================///

def main():
    inverte(nome_video)

if __name__ == "__main__":
    main()
