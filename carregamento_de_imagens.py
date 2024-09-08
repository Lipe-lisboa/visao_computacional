import cv2
from pathlib import Path

local = Path(__file__).parent
local_imgs = local.joinpath('imgs')
local_cascade = local.joinpath('cascade')

#carrega a imagem:
imagem = cv2.imread(local_imgs.joinpath('people1.jpg'))

print(f'tamanho original: {imagem.shape}')
imagem = cv2.resize(imagem,(800,600))
print(f'tamanho reduzido: {imagem.shape}')

#converte a imagem para o cinza pq tem menos pixels (mais rapido de fazer o processamento )
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
print(f'Apenas um canal: {imagem.shape}')

#abrindo a imagem:

def abrir_imagem(img):
    cv2.imshow("imagem", img)
    cv2.waitKey(0)  # Espera até que qualquer tecla seja pressionada
    cv2.destroyAllWindows()  # Fecha todas as janelas abertas


#detecção de faces:
detector_facial = cv2.CascadeClassifier(local_cascade.joinpath('haarcascade_frontalface_default.xml'))
deteccoes = detector_facial.detectMultiScale(imagem_cinza)

# (x,y,largura,altura)
print(deteccoes)
print(f'quantidade de faces encontradas: {len(deteccoes)}')

#desenhando um retangulo ao redor da face
for x,y,w,h in deteccoes:
    cv2.rectangle(imagem, (x,y), (x + w,y +h), (0,255,255), 5)

abrir_imagem(imagem)