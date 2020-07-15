########################################################################################################################
# DATA: 02/07/2020
# ALUNO: FELIPE PEREIRA CARDOSO
# E-MAIL: ffulipe@gmail.com
# GITHUB: FFULIPE
# DISCIPLINA: VISÃO COMPUTACIONAL NO MELHORAMENTO DE PLANTAS
# PROFESSOR: VINÍCIUS QUINTÃO CARNEIRO
########################################################################################################################
import numpy as np
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
#'''
# REO 01 - LISTA DE EXERCÍCIOS

# EXERCÍCIO 01:
# a) Declare os valores 43.5,150.30,17,28,35,79,20,99.07,15 como um array numpy.
print('Exercicio 1 - A')
print('-'*112)

x = np.array([43.5,150.30,17,28,35,79,20,99.07,15])
print ("Vetor: "+str(x))
print('-='*56)
# b) Obtenha as informações de dimensão, média, máximo, mínimo e variância deste vetor.
print('Exercicio 1 - B')
print('-'*112)
dim = len(x)
media = np.mean(x)
max = np.max(x)
min = np.min(x)
var = np.var(x)
print ("Dimensão: "+str(dim))
print ("Média: "+str(media))
print ("Máximo: "+str(max))
print ("Mínimo: "+str(min))
print ("Variância: "+str(var))
print('-='*56)

# c) Obtenha um novo vetor em que cada elemento é dado pelo quadrado da diferença entre cada elemento do vetor declarado
# na letra a e o valor da média deste.
print('Exercicio 1 - C')
print('-'*112)
#Poderia ter feito pelo for, porem não é tão eficiente
#'''
'''
novo_vetor = np.zeros(dim)
for i in range(dim):
    novo_vetor[i] = np.array((x[i] - np.mean(x))**2)
print ("Novo Vetor: " +str (novo_vetor))
'''
#'''
#assim é mais eficiente
new_vector =  (x - media)
print(new_vector**2)


print('-='*56)
# d) Obtenha um novo vetor que contenha todos os valores superiores a 30.
print('Exercicio 1 - D')
print('-'*112)
bool_maior_30 = x>30
print('Vetor original: ' + str(x))
print('Vetor booleano maiores que 30: ' + str(bool_maior_30))
vetor_maior_30 = x[bool_maior_30]
print('Vetor com os valores maiores que 30: ' + str(vetor_maior_30))
print('-='*56)

# e) Identifique quais as posições do vetor original possuem valores superiores a 30
print('Exercicio 1 - E')
print('-'*112)
print('Vetor original: ' + str(x))
pos_maior_30 = np.where(x>30)
print('Posições com valores maiores que 30: ' + str(pos_maior_30[0]))
print('-='*56)

# f) Apresente um vetor que contenha os valores da primeira, quinta e última posição.
print('Exercicio 1 - F')
print('-'*112)
vetor_f = np.array([x[0],x[4],x[8]])
print ("Valores da primeira, quinta e última posição: " +str(vetor_f))
print('-='*56)

# g) Crie uma estrutura de repetição usando o for para apresentar cada valor e a sua respectiva posição durante as iterações
print('Exercicio 1 - G')
print('-'*112)
for i in range(dim):
    pos = i+1
    print ("Posição: " +str(pos) + ("| Valor: " +str(x[i])))
print('-='*56)

# h) Crie uma estrutura de repetição usando o for para fazer a soma dos quadrados de cada valor do vetor.
print('Exercicio 1 - H')
print('-'*112)
sdq = np.zeros(dim)
somador=0
for i in range(dim):
    sdq[i] = (x[i])**2
    somador += sdq[i]
print ("Vetor com cada vetor ao quadrado: "+str(sdq))
print ("Soma de quadrados: " +str (somador))
print('-='*56)
# i) Crie uma estrutura de repetição usando o while para apresentar todos os valores do vetor
print('Exercicio 1 - I')
print('-'*112)
pos = 0
while x[pos]!=10:
    print(x[pos])
    pos = pos+1
    if pos==(len(x)):
       print('Posição igual a: ' +str(pos)+ ' - A condição estabelecida retornou true, vamos sair do while')
       break
print('-='*56)
# j) Crie um sequência de valores com mesmo tamanho do vetor original e que inicie em 1 e o passo seja também 1.
print('Exercicio 1 - J')
print('-'*112)
novo_vetor = np.arange(1,dim+1,1)
print('ARANGE:Sequencia 1 até 9 (passo: 1)')
print (novo_vetor)
print('-='*56)
# k) Concatene o vetor da letra a com o vetor da letra j.

print('Exercicio 1 - K')
print('-'*112)
vet_concat = np.concatenate((x,novo_vetor))
print('Vetor Concatenado')
print(vet_concat)
print('-='*56)
print('-'*112)


#'''
########################################################################################################################
########################################################################################################################
########################################################################################################################

# Exercício 02
#a) Declare a matriz abaixo com a biblioteca numpy.
# 1 3 22
# 2 8 18
# 3 4 22
# 4 1 23
# 5 2 52
# 6 2 18
# 7 2 25
print('Exercicio 2 - B')
print('-'*112)

matriz = np.array([[1,3,22],[2,8,18],[3,4,22],[4,1,23],[5,2,52],[6,2,18],[7,2,25]])
print ("Matriz: ")
print (matriz)
print('-='*56)

# b) Obtenha o número de linhas e de colunas desta matriz
print('Exercicio 2 - A')
print('-'*112)
nl,nc = np.shape(matriz)
print('Número de linhas: ' + str(nl))
print('Número de colunas: ' + str(nc))
print('-='*56)

# c) Obtenha as médias das colunas 2 e 3.
print('Exercicio 2 - C')
print('-'*112)
media_col2 = np.mean(matriz[:,1])
media_col3 = np.mean(matriz[:,2])
print("Média da coluna 2: " +str (media_col2))
print("Média da coluna 3: " +str (media_col3))
print('-='*56)
# d) Obtenha as médias das linhas considerando somente as colunas 2 e 3
print('Exercicio 2 - D')
print('-'*112)
sub_matriz = matriz[0:,1:] #a partir da linha 1 e apartir da coluna 2 ([início:fim:incremento])
medias_lin = np.mean(sub_matriz,axis=1)
print('Média Linhas: '+ str(medias_lin))
print('-='*56)

# e) Considerando que a primeira coluna seja a identificação de genótipos, a segunda nota de severidade de uma doença e
# e a terceira peso de 100 grãos. Obtenha os genótipos que possuem nota de severidade inferior a 5.
print('Exercicio 2 - E')
print('-'*112)
gen_menor_5 = np.squeeze (np.asarray(matriz[:,1]))<5 #gera um array boleano com os valores da coluna 2 <5
print("Genótipos que possuem nota de severidade inferior a 5:")
print(matriz[gen_menor_5,:])
print('-='*56)
# f) Considerando que a primeira coluna seja a identificação de genótipos, a segunda nota de severidade de uma doença e
# e a terceira peso de 100 grãos. Obtenha os genótipos que possuem nota de peso de 100 grãos superior ou igual a 22.
print('Exercicio 2 - F')
print('-'*112)
gen_maior_22 = np.squeeze (np.asarray(matriz[:,2]))>=22 #gera um array boleano com os valores da coluna 3 >=22
print("Genótipos que possuem peso de 100 grãos superior ou igual a 22:")
print(matriz[gen_maior_22,:])
print('-='*56)
# g) Considerando que a primeira coluna seja a identificação de genótipos, a segunda nota de severidade de uma doença e
# e a terceira peso de 100 grãos. Obtenha os genótipos que possuem nota de severidade igual ou inferior a 3 e peso de 100
# grãos igual ou superior a 22.
print('Exercicio 2 - G')
print('-'*112)
mask = ((matriz[:,1]<5) & (matriz[:,2]>=22))
print("Genótipos que possuem nota de severidade igual ou inferior a 3 e peso de 100 grãos igual ou superior a 22:")
print(matriz[mask,:])
print('-='*56)
# h) Crie uma estrutura de repetição com uso do for (loop) para apresentar na tela cada uma das posições da matriz e o seu
#  respectivo valor. Utilize um iterador para mostrar ao usuário quantas vezes está sendo repetido.
#  Apresente a seguinte mensagem a cada iteração "Na linha X e na coluna Y ocorre o valor: Z".
#  Nesta estrutura crie uma lista que armazene os genótipos com peso de 100 grãos igual ou superior a 25
print('Exercicio 2 - H')
print('-'*112)
contador = 0
genotipos = []
for i in np.arange(0,nl,1):
    if matriz[i, 2] >= 25:
        genotipos.append(matriz[i, 0])
    for j in np.arange(0,nc,1):
        contador += 1
        print('Iteração: '+ str(contador))
        #como em numpy a linha e coluna se inicia por 0, somei +1 ao indexador pra facilitar a visualização
        print('Na linha ' + str(i+1) + ' e na coluna ' + str(j+1) + ' ocorre o valor: ' + str(matriz[int(i),int(j)]))
        print('-'*112)
print ("Lista dos genótipos com peso de 100 grãos igual ou superior a 25")
print (genotipos)
print('-='*56)
########################################################################################################################
########################################################################################################################
########################################################################################################################

# EXERCÍCIO 03:
# a) Crie uma função em um arquivo externo (outro arquivo .py) para calcular a média e a variância amostral um vetor qualquer,
# baseada em um loop (for).
print('Exercicio 3 - A')
print('-'*112)
from funcoes_ex3_felipe import media, var_amostral
print('Exemplo da função funcionando:')
vet = np.array([4,8,3,9,7,5])
print('Vetor: ' +str(vet)+' Média: ' +str (media(vet)) + ' Variância amostral: '+str (var_amostral(vet)))
print('-='*56)
# b) Simule três arrays com a biblioteca numpy de 10, 100, e 1000 valores e com distribuição normal com média 100 e variância 2500.
# Pesquise na documentação do numpy por funções de simulação.
print('Exercicio 3 - B')
print('-'*112)
med, sigma = 100, 50
vetor1 = np.random.normal(med,sigma,10)   #np.random.normal (media, desvio padrao, tamanho)
vetor2 = np.random.normal(med,sigma,100)
vetor3 = np.random.normal(med,sigma,1000)
vetor4 = np.random.normal(med,sigma,100000)
#se quiser que apareça as simulações retirar as aspas abaixo
"""
print('Vetor 1 - 10 valores')
print(vetor1)
print('-'*112)
print('Vetor 2 - 100 valores')
print(vetor2)
print('-'*112)
print('Vetor 3 - 1000 valores')
print(vetor3)
print('-='*56)
"""
# c) Utilize a função criada na letra a para obter as médias e variâncias dos vetores simulados na letra b.
print('Exercicio 3 - C')
print('-'*112)
print('Vetor 1 - 10 valores')
print('Média: ' +str (media(vetor1)) + ' Variância amostral: '+str (var_amostral(vetor1)))
print('-'*112)
print('Vetor 2 - 100 valores')
print('Média: ' +str (media(vetor2)) + ' Variância amostral: '+str (var_amostral(vetor2)))
print('-'*112)
print('Vetor 3 - 1000 valores')
print('Média: ' +str (media(vetor3)) + ' Variância amostral: '+str (var_amostral(vetor3)))
print('-'*112)
print('Vetor 4 - 100000 valores')
print('Média: ' +str (media(vetor3)) + ' Variância amostral: '+str (var_amostral(vetor3)))
print('-='*56)
# d) Crie histogramas com a biblioteca matplotlib dos vetores simulados com valores de 10, 100, 1000 e 100000.
from matplotlib import pyplot as plt
plt.style.use('seaborn-muted')

count, bins, ignored = plt.hist(vetor1, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
np.exp( - (bins - med)**2 / (2 * sigma**2) ),linewidth=2, color='r')
plt.title('Vetor 1 - 10 valores')
plt.show()
count, bins, ignored = plt.hist(vetor2, 100, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
np.exp( - (bins - med)**2 / (2 * sigma**2) ),linewidth=2, color='r')
plt.title('Vetor 2 - 100 valores')
plt.show()
count, bins, ignored = plt.hist(vetor3, 100, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
np.exp( - (bins - med)**2 / (2 * sigma**2) ),linewidth=2, color='r')
plt.title('Vetor 3 - 1000 valores')
plt.show()
count, bins, ignored = plt.hist(vetor4, 100, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
np.exp( - (bins - med)**2 / (2 * sigma**2) ),linewidth=2, color='r')
plt.title('Vetor 4 - 100000 valores')
plt.show()

########################################################################################################################
########################################################################################################################
########################################################################################################################

# EXERCÍCIO 04:
# a) O arquivo dados.txt contem a avaliação de genótipos (primeira coluna) em repetições (segunda coluna) quanto a quatro
# variáveis (terceira coluna em diante). Portanto, carregue o arquivo dados.txt com a biblioteca numpy, apresente os dados e
# obtenha as informaçõesde dimensão desta matriz.
print('Exercicio 4 - A')
print('-'*112)
dados = np.loadtxt('dados.txt')
print('Dados')
print(dados)
# Obtendo a dimensão de uma matriz
nl,nc = np.shape(dados)
print('Número de linhas: ' + str(nl))
print('Número de colunas: ' + str(nc))
print('-='*56)
# b) Pesquise sobre as funções np.unique e np.where da biblioteca numpy
'''
help (np.unique)
help (np.where)
'''
# c) Obtenha de forma automática os genótipos e quantas repetições foram avaliadas
print('Exercicio 4 - C')
print('-'*112)
print ('GENÓTIPOS: ')
genotipos = np.unique(dados[0:30,0:1], axis=0)
nlg,ncg = np.shape(genotipos)
print('Número de linhas: ' + str(nl))
print('Número de colunas: ' + str(nc))
print (np.unique(dados[0:30,0:1], axis=0)) #dados[0:30,0:1] ([início:fim:incremento])
print ('REPETIÇÕES: ')
print (np.unique(dados[0:30,1:2], axis=0))
print('-='*56)
# d) Apresente uma matriz contendo somente as colunas 1, 2 e 4
print('Exercicio 4 - D')
print('-'*112)
print('Matriz coluna 1, 2 e 4')
sub_dados =  dados[:,[0,1,3]]
print (sub_dados)
# e) Obtenha uma matriz que contenha o máximo, o mínimo, a média e a variância de cada genótipo para a variavel da coluna 4.
# Salve esta matriz em bloco de notas.
print('Exercicio 4 - E')
print('-'*112)

minimos = np.zeros((nlg,1))
maximos = np.zeros((nlg,1))
medias = np.zeros((nlg,1))
vars = np.zeros((nlg,1))
it=0
for i in np.arange(0,nl,3): #percorre as 30 linhas do vetor original de acordo com o numero de repetições

    minimos[it,0] = np.min(sub_dados[i:i + 3, 2], axis=0)
    maximos[it,0] = np.max(sub_dados[i:i + 3, 2], axis=0)
    medias[it,0] = np.mean(sub_dados[i:i + 3, 2], axis=0)
    vars[it,0] = np.var(sub_dados[i:i + 3, 2], axis=0)
    it += 1 #incrementa + 1 no it

print('Genótipos     Min     Max      Média    Variância')
matriz_concat = np.concatenate((genotipos,minimos,maximos,medias,vars),axis=1)
print (matriz_concat)
#help (np.savetxt)
np.savetxt('matriz_ex3.txt', matriz_concat, delimiter=' ')
# f) Obtenha os genótipos que possuem média (médias das repetições) igual ou superior a 500 da matriz gerada na letra anterior.
print('Exercicio 4 - F')
print('-'*112)
dados2 = np.loadtxt('matriz_ex3.txt')
gen_maior_500 = np.squeeze (np.asarray(dados2[:,3]))>=500 #gera um array boleano com os valores da coluna 4 >=500
print("Genótipos que possuem média maior ou igual a 500:")
print(dados2[gen_maior_500,0])
# g) Apresente os seguintes graficos:
#    - Médias dos genótipos para cada variável. Utilizar o comando plt.subplot para mostrar mais de um grafico por figura
#    - Disperão 2D da médias dos genótipos (Utilizar as três primeiras variáveis). No eixo X uma variável e no eixo Y outra.
dados = np.loadtxt('dados.txt')
media1 = np.zeros((nlg,1))
media2 = np.zeros((nlg,1))
media3 = np.zeros((nlg,1))
media4 = np.zeros((nlg,1))
media5 = np.zeros((nlg,1))
it=0
for i in np.arange(0,30,3): #percorre as 30 linhas do vetor original de acordo com o numero de repetições
    media1[it,0] = np.mean(dados[i:i + 3, 2], axis=0)
    media2[it,0] = np.mean(dados[i:i + 3, 3], axis=0)
    media3[it,0] = np.mean(dados[i:i + 3, 4], axis=0)
    media4[it,0] = np.mean(dados[i:i + 3, 5], axis=0)
    media5[it,0] = np.mean(dados[i:i + 3, 6], axis=0)
    it += 1 #incrementa + 1 no it

dados_medias = np.concatenate((genotipos,media1,media2,media3,media4,media5),axis=1) #matriz com as medias dos genotipos p as 5 var
nl,nc = np.shape(dados_medias)
#gerando gráfico barras das médias
plt.style.use('ggplot')
plt.figure('Gráfico Médias')
plt.subplot(2,3,1) #a figura tem 2linhas, 3 colunas, e esse grafico vai ocupar a posição 1
plt.bar(dados_medias[:,0],dados_medias[:,1])
plt.title('Variável 1')
plt.xticks(dados_medias[:,0])

plt.subplot(2,3,2)
plt.bar(dados_medias[:,0],dados_medias[:,2])
plt.title('Variável 2')
plt.xticks(dados_medias[:,0])

plt.subplot(2,3,3)
plt.bar(dados_medias[:,0],dados_medias[:,3])
plt.title('Variável 3')
plt.xticks(dados_medias[:,0])

plt.subplot(2,3,4)
plt.bar(dados_medias[:,0],dados_medias[:,4])
plt.title('Variável 4')
plt.xticks(dados_medias[:,0])

plt.subplot(2,3,5)
plt.bar(dados_medias[:,0],dados_medias[:,5])
plt.title('Variável 5')
plt.xticks(dados_medias[:,0])
plt.show()

#disperssão

plt.style.use('ggplot')
fig = plt.figure('Disperão 2D da médias dos genótipos')
plt.subplot(2,2,1)
cores = ['black','blue','red','green','yellow','pink','cyan','orange','darkviolet','slategray']

for i in np.arange(0,nl,1):
    plt.scatter(dados_medias[i,1], dados_medias[i,2],s=50,alpha=0.8,label = dados_medias[i,0],c = cores[i])

plt.xlabel('Var 1')
plt.ylabel('Var 2')
plt.subplot(2,2,2)
for i in np.arange(0,nl,1):
    plt.scatter(dados_medias[i,2], dados_medias[i,3],s=50,alpha=0.8,label = dados_medias[i,0],c = cores[i])

plt.xlabel('Var 2')
plt.ylabel('Var 3')
plt.subplot(2,2,3)
for i in np.arange(0,nl,1):
    plt.scatter(dados_medias[i,1], dados_medias[i,3],s=50,alpha=0.8,label = dados_medias[i,0],c = cores[i])

plt.xlabel('Var 1')
plt.ylabel('Var 3')
plt.legend(bbox_to_anchor=(2.08, 0.7), title='Genotipos', borderaxespad=0., ncol=5)
plt.show()

########################################################################################################################
########################################################################################################################
########################################################################################################################