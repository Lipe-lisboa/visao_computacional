import cv2
from pathlib import Path

local = Path(__file__).parent
local_imgs = local.joinpath('imgs')
local_cascade = local.joinpath('cascade')

imagem = cv2.imread(local_imgs.joinpath('car.jpg'))
imagem = cv2.resize(imagem, (800,600))
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

def abrir_imagem(img):
    cv2.imshow("imagem", img)
    cv2.waitKey(0)  # Espera até que qualquer tecla seja pressionada
    cv2.destroyAllWindows()  # Fecha todas as janelas abertas

detector_carros = cv2.CascadeClassifier(local_cascade.joinpath('cars.xml'))

deteccoes = detector_carros.detectMultiScale(
    imagem_cinza,
    scaleFactor=1.03,
    minNeighbors= 5,
    minSize=(35,35),
    #maxSize=(27,27),
)

for x,y,w,h in deteccoes:
    cv2.rectangle(imagem, (x,y), (x + w,y +h), (0,0,255), 2)

print(deteccoes)
abrir_imagem(imagem)
