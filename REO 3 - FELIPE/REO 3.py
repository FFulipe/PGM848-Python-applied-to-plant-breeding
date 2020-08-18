########################################################################################################################
# DATA: 29/07/2020
# ALUNO: FELIPE PEREIRA CARDOSO
# E-MAIL: ffulipe@gmail.com
# GITHUB: FFULIPE
# DISCIPLINA: VISÃO COMPUTACIONAL NO MELHORAMENTO DE PLANTAS
# PROFESSOR: VINÍCIUS QUINTÃO CARNEIRO
########################################################################################################################
import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt


#EXERCÍCIO 01:
#a) Aplique o filtro de média com cinco diferentes tamanhos de kernel e compare os resultados com a imagem original;
# Leitura da imagem
nome_arquivo = '104.ge'
img_bgr = cv2.imread(nome_arquivo,1)
img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
r,g,b = cv2.split(img_rgb)
# Filtros
# Média
img_fm_1 = cv2.blur(img_rgb,(3,3))
img_fm_2 = cv2.blur(img_rgb,(5,5))
img_fm_3 = cv2.blur(img_rgb,(7,7))
img_fm_4 = cv2.blur(img_rgb,(9,9))
img_fm_5 = cv2.blur(img_rgb,(11,11))
# Apresentar imagens no matplotlib
plt.figure('Filtros')
plt.subplot(2,3,1)
plt.imshow(img_rgb)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("ORIGINAL")

plt.subplot(2,3,2)
plt.imshow(img_fm_1)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("3x3")

plt.subplot(2,3,3)
plt.imshow(img_fm_2)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("5x5")

plt.subplot(2,3,4)
plt.imshow(img_fm_3)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("7x7")

plt.subplot(2,3,5)
plt.imshow(img_fm_4)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("9x9")

plt.subplot(2,3,6)
plt.imshow(img_fm_5)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("11x11")

plt.show()

#b) Aplique diferentes tipos de filtros com pelo menos dois tamanhos de kernel e compare os
# resultados entre si e com a imagem original.

#Filtros
img_filtro_media1 = cv2.blur(img_rgb,(9,9))
img_filtro_media2 = cv2.blur(img_rgb,(15,15))

img_filtro_gaussiano1 = cv2.GaussianBlur(img_rgb,(9,9),0) # Média Ponderada
img_filtro_gaussiano2 = cv2.GaussianBlur(img_rgb,(11,11),0) # Média Ponderada

img_filtro_mediana1 = cv2.medianBlur(img_rgb,9)
img_filtro_mediana2 = cv2.medianBlur(img_rgb,11)

img_filtro_bilateral1= cv2.bilateralFilter(img_rgb,9,9,33)
img_filtro_bilateral2= cv2.bilateralFilter(img_rgb,11,11,55)

# Apresentar imagens no matplotlib
plt.figure('Filtros 9x9')
plt.subplot(2,3,1)
plt.imshow(img_rgb)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("ORIGINAL")

plt.subplot(2,3,2)
plt.imshow(img_filtro_media1)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("Média")

plt.subplot(2,3,3)
plt.imshow(img_filtro_gaussiano1)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("Gaussiano")

plt.subplot(2,3,4)
plt.imshow(img_filtro_mediana1)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("Mediana")

plt.subplot(2,3,5)
plt.imshow(img_filtro_bilateral1)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("Bilateral")

plt.show()

plt.figure('Filtros 11x11')
plt.subplot(2,3,1)
plt.imshow(img_rgb)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("ORIGINAL")

plt.subplot(2,3,2)
plt.imshow(img_filtro_media2)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("Média")

plt.subplot(2,3,3)
plt.imshow(img_filtro_gaussiano2)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("Gaussiano")

plt.subplot(2,3,4)
plt.imshow(img_filtro_mediana2)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("Mediana")

plt.subplot(2,3,5)
plt.imshow(img_filtro_bilateral2)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("Bilateral")

plt.show()
#c) Realize a segmentação da imagem utilizando o processo de limiarização. Utilizando o
# reconhecimento de contornos, identifique e salve os objetos de interesse. Além disso, acesse
# as bibliotecas Opencv e Scikit-Image, verifique as variáveis que podem ser mensuradas e
# extraia as informações pertinentes (crie e salve uma tabela com estes dados). Apresente todas
# as imagens obtidas ao longo deste processo.

r_bilateral = cv2.bilateralFilter(r,15,15,55)
g_bilateral = cv2.bilateralFilter(g,15,15,55)
b_bilateral = cv2.bilateralFilter(b,15,15,55)
# Apresentar imagens no matplotlib
plt.figure('Filtro Bilateral')
plt.subplot(2,3,1)
plt.imshow(r,cmap='gray')
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("R")

plt.subplot(2,3,2)
plt.imshow(g,cmap='gray')
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("G")

plt.subplot(2,3,3)
plt.imshow(b,cmap='gray')
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("B")

plt.subplot(2,3,4)
plt.imshow(r_bilateral,cmap='gray')
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("R Bilateral")

plt.subplot(2,3,5)
plt.imshow(g_bilateral,cmap='gray')
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("G Bilateral")

plt.subplot(2,3,6)
plt.imshow(b_bilateral,cmap='gray')
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("B Bilateral")
plt.show()
#Segmentação utilizando o Filtro Bilateral no canal R

hist_r_bilateral = cv2.calcHist([r_bilateral],[0], None, [256],[0,256])
l_f,img_l_f = cv2.threshold(r_bilateral,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.figure('Segmentacao')
plt.subplot(2,2,1)
plt.imshow(r_bilateral,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Canal R - Filtro Bilateral')

plt.subplot(2,2,4)
plt.plot(hist_r_bilateral)
plt.axvline(x=l_f,color = 'r')
plt.title("L: "+str(l_f))
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(2,2,2)
plt.imshow(img_l_f,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Máscara')

plt.show()
#obtendo a imagem segmentada
img_segmentada = cv2.bitwise_and(img_rgb,img_rgb,mask=img_l_f)
plt.figure('Segmentada')
plt.imshow(img_segmentada)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("Segmentada")
plt.show()

# Objetos
mascara = img_l_f.copy() #cópia pra não modificar a mascara original
cnts,h = cv2.findContours(mascara, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) #cv2.CHAIN_APPROX_SIMPLE é a forma mais simples de encontrar os contornos

# Dados Sementes
from skimage.measure import regionprops
print('Dados Grãos')
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_properties/py_contour_properties.html
ng = len(cnts)
sementes = np.zeros((ng,1))
dimensao = np.zeros((ng,1))
eixo_menor= np.zeros((ng,1))
eixo_maior=np.zeros((ng,1))
razao = np.zeros((ng,1))
area_1 = np.zeros((ng,1))
perimetro = np.zeros((ng,1))

print (ng)
for (i, c) in enumerate(cnts):

	(x, y, w, h) = cv2.boundingRect(c)  #cria retangulos em volta do contorno x = posição inicio no eixo x, y = inicio no eixo y; w=largura;h=altura
	obj = img_l_f[y:y+h,x:x+w] #recortando os graos
	obj_rgb = img_segmentada[y:y+h,x:x+w]
	obj_bgr = cv2.cvtColor(obj_rgb,cv2.COLOR_RGB2BGR)
	cv2.imwrite('graos/s'+str(i+1)+'.png',obj_bgr)
	#cv2.imwrite('graos/sb'+str(i+1)+'.png',obj)
    #regionprops solicita a imagem binária
	regiao = regionprops(obj) #https: // scikit - image.org / docs / dev / api / skimage.measure.html  # skimage.measure.regionprops
	print('Semente: ', str(i+1))
	sementes[i,0] = i+1
	print('Dimensão da Imagem: ', np.shape(obj))
	#dim = np.shape(obj)
	#dimensao[0, i] = dim
	print('Medidas Físicas')
	print('Centroide: ', regiao[0].centroid)
	print('Comprimento do eixo menor: ', regiao[0].minor_axis_length)
	eixo_menor[i,0] = regiao[0].minor_axis_length
	print('Comprimento do eixo maior: ', regiao[0].major_axis_length)
	eixo_maior[i,0] = regiao[0].major_axis_length
	print('Razão: ', regiao[0].major_axis_length / regiao[0].minor_axis_length)
	razao[i,0] = regiao[0].major_axis_length / regiao[0].minor_axis_length
	# contourArea solicita o contorno
	area = cv2.contourArea(c)
	print('Área: ', area)
	print('Perímetro: ', cv2.arcLength(c,True))
	area_1 [i,0] = area
	perimetro [i,0] = cv2.arcLength(c,True)

	print('Medidas de Cor')
	min_val_r, max_val_r, min_loc_r, max_loc_r = cv2.minMaxLoc(obj_rgb[:,:,0], mask=obj)
	print('Valor Mínimo no R: ', min_val_r, ' - Posição: ', min_loc_r)
	print('Valor Máximo no R: ', max_val_r, ' - Posição: ', max_loc_r)
	med_val_r = cv2.mean(obj_rgb[:,:,0], mask=obj)
	print('Média no Vermelho:  %.2f' %med_val_r[0])


	min_val_g, max_val_g, min_loc_g, max_loc_g = cv2.minMaxLoc(obj_rgb[:, :, 1], mask=obj)
	print('Valor Mínimo no G: ', min_val_g, ' - Posição: ', min_loc_g)
	print('Valor Máximo no G: ', max_val_g, ' - Posição: ', max_loc_g)
	med_val_g = cv2.mean(obj_rgb[:,:,1], mask=obj)
	print('Média no Verde:  %.2f' %med_val_g[0])

	min_val_b, max_val_b, min_loc_b, max_loc_b = cv2.minMaxLoc(obj_rgb[:, :, 2], mask=obj)
	print('Valor Mínimo no B: ', min_val_b, ' - Posição: ', min_loc_b)
	print('Valor Máximo no B: ', max_val_b, ' - Posição: ', max_loc_b)
	med_val_b = cv2.mean(obj_rgb[:,:,2], mask=obj)
	print('Média no Azul: %.2f'  %med_val_b[0])
	print('-'*50)
print('Total de sementes: ', len(cnts))
print('-'*50)
data = np.concatenate((sementes,eixo_menor,eixo_maior,razao,area_1,perimetro),axis=1)


#Criando o DataFrame
df = pd.DataFrame(data, columns=['Semente','Eixo Menor','Eixo Maior','Razao','Area','Perimetro'])
df.set_index('Semente', inplace=True)
print(df)
df.to_csv('tabela.csv')

seg = img_segmentada.copy()
cv2.drawContours(seg,cnts,-1,(0,255,0),2)

plt.figure('Sementes')
plt.subplot(1,2,1)
plt.imshow(seg)
plt.xticks([])
plt.yticks([])
plt.title('Arroz GroundEye')

plt.subplot(1,2,2)
plt.imshow(obj_rgb)
plt.xticks([])
plt.yticks([])
plt.title('Grão')
plt.show()

#Utilizando máscaras, apresente o histograma somente dos objetos de interesse.
s82 = 'graos/s82.png'
s82_bgr = cv2.imread(s82, 1)
s82 = cv2.cvtColor(s82_bgr, cv2.COLOR_BGR2RGB)
sb82 = "graos/sb82.png"
sb82_bgr = cv2.imread(sb82, 0)
hist_segmentada_R_1 = cv2.calcHist([s82],[0], sb82_bgr, [256],[0,256])
hist_segmentada_G_1 = cv2.calcHist([s82],[1], sb82_bgr, [256],[0,256])
hist_segmentada_B_1 = cv2.calcHist([s82],[2], sb82_bgr, [256],[0,256])

s32 = 'graos/s32.png'
s32_bgr = cv2.imread(s32, 1)
s32 = cv2.cvtColor(s32_bgr, cv2.COLOR_BGR2RGB)
sb32 = "graos/sb32.png"
sb32_bgr = cv2.imread(sb32, 0)
hist_segmentada_R_2 = cv2.calcHist([s32],[0], sb32_bgr, [256],[0,256])
hist_segmentada_G_2 = cv2.calcHist([s32],[1], sb32_bgr, [256],[0,256])
hist_segmentada_B_2 = cv2.calcHist([s32],[2], sb32_bgr, [256],[0,256])

plt.figure('Questão 1.d-1/2')
plt.subplot(3,3,2)
plt.imshow(s82)
plt.title('Objeto: 1 - s82')

plt.subplot(3, 3, 4)
plt.imshow(s82[:,:,0],cmap='gray')
plt.title('Objeto: 1 - R')

plt.subplot(3, 3, 5)
plt.imshow(s82[:,:,1],cmap='gray')
plt.title('Objeto: 1 - G')

plt.subplot(3, 3, 6)
plt.imshow(s82[:,:,2],cmap='gray')
plt.title('Objeto: 1 - B')

plt.subplot(3, 3, 7)
plt.plot(hist_segmentada_R_1, color='r')
plt.title("Histograma - R")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3, 3, 8)
plt.plot(hist_segmentada_G_1, color='g')
plt.title("Histograma - G")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3, 3, 9)
plt.plot(hist_segmentada_B_1, color='b')
plt.title("Histograma - B")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")
plt.show()

plt.figure('Questão 1.d-2/2')
plt.subplot(3,3,2)
plt.imshow(s32)
plt.title('Objeto: 2 - s32')

plt.subplot(3, 3, 4)
plt.imshow(s32[:,:,0],cmap='gray')
plt.title('Objeto: 2 - R')

plt.subplot(3, 3, 5)
plt.imshow(s32[:,:,1],cmap='gray')
plt.title('Objeto: 2 - G')

plt.subplot(3, 3, 6)
plt.imshow(s32[:,:,2],cmap='gray')
plt.title('Objeto: 2 - B')

plt.subplot(3, 3, 7)
plt.plot(hist_segmentada_R_2, color='r')
plt.title("Histograma - R")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3, 3, 8)
plt.plot(hist_segmentada_G_2, color='g')
plt.title("Histograma - G")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3, 3, 9)
plt.plot(hist_segmentada_B_2, color='b')
plt.title("Histograma - B")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")
plt.show()
print('-'*50)
print(' ')
#e) Realize a segmentação da imagem utilizando a técnica de k-means. Apresente as imagens obtidas neste processo.

print('Dimensão: ',np.shape(img_rgb))
print(np.shape(img_rgb)[0], ' x ',np.shape(img_rgb)[1], ' = ', np.shape(img_rgb)[0] * np.shape(img_rgb)[1])
print('-'*80)

pixel_values = img_rgb.reshape((-1, 3))
pixel_values = np.float32(pixel_values)
print('-'*80)
print('Dimensão Matriz: ',pixel_values.shape)
print('-'*80)

# K-means
# Critério de Parada
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.5)
# Número de Grupos (k)
k = 2
dist, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
print('-'*80)
print('SQ das Distâncias de Cada Ponto ao Centro: ', dist)
print('-'*80)
print('Dimensão labels: ', labels.shape)
print('Valores únicos: ',np.unique(labels))
print('Tipo labels: ', type(labels))

# flatten the labels array
labels = labels.flatten()
print('-'*80)
print('Dimensão flatten labels: ', labels.shape)
print('Tipo labels (f): ', type(labels))
print('-'*80)

# Valores dos labels
val_unicos,contagens = np.unique(labels,return_counts=True)
val_unicos = np.reshape(val_unicos,(len(val_unicos),1))
contagens = np.reshape(contagens,(len(contagens),1))
hist = np.concatenate((val_unicos,contagens),axis=1)
print('Histograma')
print(hist)
print('-'*80)
print('Centroides Decimais')
print(centers)
print('-'*80)

# Conversão dos centroides para valores de interos de 8 digitos
centers = np.uint8(centers)
print('-'*80)
print('Centroides uint8')
print(centers)
print('-'*80)

# Conversão dos pixels para a cor dos centroides
matriz_segmentada = centers[labels]
print('-'*80)
print('Dimensão Matriz Segmentada: ',matriz_segmentada.shape)
print('Matriz Segmentada')
print(matriz_segmentada[0:5,:])
print('-'*80)

# Reformatar a matriz na imagem de formato original
img_segmentada = matriz_segmentada.reshape(img_rgb.shape)

# Grupo 1
original_01 = np.copy(img_rgb)
matriz_or_01 = original_01.reshape((-1, 3))
matriz_or_01[labels != 1] = [0, 0, 0]
img_final_01 = matriz_or_01.reshape(img_rgb.shape)

# Grupo 2
original_02 = np.copy(img_rgb)
matriz_or_02 = original_02.reshape((-1, 3))
matriz_or_02[labels == 1] = [0, 0, 0]
img_final_02 = matriz_or_02.reshape(img_rgb.shape)
########################################################################################################################

# Apresentar Imagem
plt.figure('Questão 1.e')
plt.subplot(2,2,1)
plt.imshow(img_rgb)
plt.title('Imagem RGB')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,2)
plt.imshow(img_segmentada)
plt.title('Rótulos')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,3)
plt.imshow(img_final_01)
plt.title('Grupo 1')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,4)
plt.imshow(img_final_02)
plt.title('Grupo 2')
plt.xticks([])
plt.yticks([])
plt.show()
print('-'*50)
print(' ')


#f) Realize a segmentação da imagem utilizando a técnica de watershed. Apresente as imagens obtidas neste processo.
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
r,g,b = cv2.split(img_rgb)
limiar_1f, mask = cv2.threshold(r,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
img_dist = ndimage.distance_transform_edt(mask)
localmax = peak_local_max(img_dist, indices=False, min_distance=25, labels=mask)

print('Quantidade de Picos')
print(np.unique(localmax,return_counts=True))
print('-'*50)

marcadores,n_marcadores = ndimage.label(localmax, structure=np.ones((3, 3)))

print('Marcadores')
print(np.unique(marcadores,return_counts=True))
print('-'*50)
img_ws = watershed(-img_dist, marcadores, mask=mask)

print('Watershed')
print(np.unique(img_ws,return_counts=True))
print("Número de sementes: ", len(np.unique(img_ws)) - 1)
img_final = np.copy(img_rgb)
img_final[img_ws != 35] = [0,0,0]

plt.figure('Questão 1.f')
plt.subplot(2,3,1)
plt.imshow(img_rgb)
plt.xticks([])
plt.yticks([])
plt.title('RGB')

plt.subplot(2,3,2)
plt.imshow(r,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('R')

plt.subplot(2,3,3)
plt.imshow(mask,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Máscara binária')

plt.subplot(2,3,4)
plt.imshow(img_dist,cmap='jet')
plt.xticks([])
plt.yticks([])
plt.title('Imagem distância')

plt.subplot(2,3,5)
plt.imshow(img_ws,cmap='jet')
plt.xticks([])
plt.yticks([])
plt.title('Imagem folha segmentada')

plt.subplot(2,3,6)
plt.imshow(img_final)
plt.xticks([])
plt.yticks([])
plt.title('Seleção')
plt.show()
print('-'*50)
print(' ')


#g) Compare os resultados das três formas de segmentação (limiarização, k-means e watershed) e identifique as potencialidades de cada delas.')

plt.figure('Questão 1.g')
plt.subplot(1,3,1)
plt.imshow(mask,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('OTSU')

plt.subplot(1,3,2)
plt.imshow(img_final_01)
plt.title('K-MEANS')
plt.xticks([])
plt.yticks([])

plt.subplot(1,3,3)
plt.imshow(img_ws)
plt.xticks([])
plt.yticks([])
plt.title('Wathershed')
plt.show()
print('-'*50)
print(' ')
