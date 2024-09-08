import cv2
from pathlib import Path

local = Path(__file__).parent
local_imgs = local.joinpath('imgs')
local_cascade = local.joinpath('cascade')

#carrega a imagem:
imagem = cv2.imread(local_imgs.joinpath('people1.jpg'))

print(f'tamanho original: {imagem.shape}')
imagem = cv2.resize(imagem,(1200,800))
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
detector_olhos = cv2.CascadeClassifier(local_cascade.joinpath('haarcascade_eye.xml'))

# scaleFactor= (quanto mais proximo esta a face menor a escala, visse e versa)
deteccoes = detector_facial.detectMultiScale(
    imagem_cinza,
    scaleFactor=1.2,
    minNeighbors=3,
    minSize=(55,55),
    maxSize=(100,100)
)

#desenhando um retangulo ao redor da face
# (x,y,largura,altura)
for x,y,w,h in deteccoes:
    cv2.rectangle(imagem, (x,y), (x + w,y +h), (0,255,0), 2)

deteccoes_olhos = detector_olhos.detectMultiScale(
    imagem_cinza,
    scaleFactor=1.05,
    minNeighbors= 4,
    minSize=(20,20),
    maxSize=(27,27)
)

for x,y,w,h in deteccoes_olhos:
    cv2.rectangle(imagem, (x,y), (x + w,y +h), (0,0,255), 2)

print(deteccoes_olhos)
abrir_imagem(imagem)