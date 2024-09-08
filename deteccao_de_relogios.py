import cv2
from pathlib import Path

local = Path(__file__).parent
local_imgs = local.joinpath('imgs')
local_cascade = local.joinpath('cascade')

#carrega a imagem:
imagem = cv2.imread(local_imgs.joinpath('clock.jpg'))

print(f'tamanho original: {imagem.shape}')

#converte a imagem para o cinza pq tem menos pixels (mais rapido de fazer o processamento )
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
print(f'Apenas um canal: {imagem.shape}')

#abrindo a imagem:
def abrir_imagem(img):
    cv2.imshow("imagem", img)
    cv2.waitKey(0)  # Espera até que qualquer tecla seja pressionada
    cv2.destroyAllWindows()  # Fecha todas as janelas abertas


#detecção de faces:
detector_relogio = cv2.CascadeClassifier(local_cascade.joinpath('clocks.xml'))

#desenhando um retangulo ao redor do relogio
# (x,y,largura,altura)
deteccoes = detector_relogio.detectMultiScale(
    imagem_cinza,
    scaleFactor=1.03,
    minNeighbors= 1,
    #minSize=(20,20),
    #maxSize=(27,27)
)

for x,y,w,h in deteccoes:
    cv2.rectangle(imagem, (x,y), (x + w,y +h), (0,0,255), 2)

print(deteccoes)
abrir_imagem(imagem)