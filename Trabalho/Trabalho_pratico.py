########################################################################################################################
# DATA: 29/08/2020
# ALUNO: Alex, Ana, Felipe, Rafael e Vinicius
# E-MAIL: ffulipe@gmail.com
# GITHUB: FFULIPE
# DISCIPLINA: VISÃO COMPUTACIONAL NO MELHORAMENTO DE PLANTAS
# PROFESSOR: VINÍCIUS QUINTÃO CARNEIRO
########################################################################################################################
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.measure import regionprops
from sklearn.linear_model import
def sementes_dados (nome_arquivo):
    img_bgr = cv2.imread(nome_arquivo, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(img_rgb)
    r_bilateral = cv2.bilateralFilter(r, 15, 15, 55)
    l_f, img_l_f = cv2.threshold(r_bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_segmentada = cv2.bitwise_and(img_rgb, img_rgb, mask=img_l_f)
    mascara = img_l_f.copy()  # cópia pra não modificar a mascara original
    cnts, h = cv2.findContours(mascara, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    ng = len(cnts)
    sementes = np.zeros((ng, 1))
    eixo_menor = np.zeros((ng, 1))
    eixo_maior = np.zeros((ng, 1))
    razao = np.zeros((ng, 1))
    area_1 = np.zeros((ng, 1))
    perimetro = np.zeros((ng, 1))


    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)  # cria retangulos em volta do contorno x = posição inicio no eixo x, y = inicio no eixo y; w=largura;h=altura
        obj = img_l_f[y:y + h, x:x + w]  # recortando os graos
        obj_rgb = img_segmentada[y:y + h, x:x + w]
        obj_bgr = cv2.cvtColor(obj_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite('graos/s' + str(i + 1) + '.png', obj_bgr)
        cv2.imwrite('graos/sb'+str(i+1)+'.png',obj)
        # regionprops solicita a imagem binária
        regiao = regionprops(obj)  # https: // scikit - image.org / docs / dev / api / skimage.measure.html  # skimage.measure.regionprops
        
        sementes[i, 0] = i + 1
        
        eixo_menor[i, 0] = regiao[0].minor_axis_length
        
        eixo_maior[i, 0] = regiao[0].major_axis_length
        
        razao[i, 0] = eixo_maior[i, 0] / eixo_menor[i, 0]
        
        area = cv2.contourArea(c)
        area_1[i, 0] = area
        perimetro[i, 0] = cv2.arcLength(c, True)
    data = np.concatenate((sementes, eixo_menor, eixo_maior, razao, area_1, perimetro), axis=1)
    df = pd.DataFrame(data, columns=['Semente', 'Eixo Menor', 'Eixo Maior', 'Razao', 'Area', 'Perimetro'])
    df.set_index('Semente', inplace=True)
    return df

dados = sementes_dados('imagem/108.ge')
print(dados)
dados.to_csv('tabela108.csv')

dados_medias = np.loadtxt('dados_medias.txt')
nl,nc = np.shape(dados_medias)

x=dados_medias.iloc[i,4].values
y=dados_medias.iloc[i,5].values
correlacao=np.corrcoef(x,y)
x=x.reshape(-1,1)
regressor=()
regressor.fit(x,y)

#plt.style.use('ggplot')
fig = plt.figure('Disperão 2D da médias dos genótipos')
plt.subplot(1,2,1)
cores = ['black','blue','red','green','yellow','pink','cyan','orange','darkviolet','slategray','darksalmon','olivedrab','lightseagreen','navy','mediumvioletred','palevioletred','sienna','palegreen','tomato','indigo']
for i in np.arange(0,nl,1):
    plt.scatter(dados_medias[i,4], dados_medias[i,5],s=50,alpha=0.8,label = dados_medias[i,0],c = cores[i])
plt.plot(x,regressor.predict(x),color=red)
plt.xlabel('Área')
plt.ylabel('MMG')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Genotipos', borderaxespad=0., ncol=2)
plt.show()
