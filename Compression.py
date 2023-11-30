#Taux de compression de 15 (ni plus ni moins)
#Choix libre du type de compression (DCT/DWT)
#Pour quantifer le taux de compression, image signée sur 8bits entier donc Por=nb-px*8
#Utiliser que des niveau de gris
#C= Por/Pcomp ~15
#image 4

"""""

-définir delta du niveau selon la dynamique des histo
-quantification cours slide 49
-calcul poids? 
-décodage formule cours slide 51

"""""

import numpy as np
import pywt
import cv2
from matplotlib import pyplot as plt


def plage(L):
    """ Retourne le codage en plage de la liste L """
    i = 1
    n = 0
    res = []
    for k in L:
        if k == i:
            n += 1
        else:
            res.append(n)
            n = 1
            i = 1 - i
    res.append(n)
    return res


def un_plage(L):
    """ Retourne le décodage en plage de la liste L """
    res = []
    i = 1
    for n in L:
        res += n * [i]
        i = 1-i
    return res

def bits(i):
    """ Retourne le nombre de bits nécessaire pour encoder l'entier i """
    return int(np.log(i) / np.log(2)) + 1


def pos_encoding(L):
    """ Prend la liste des positions non nulles [1, 1, 0, 0, 1, 0, 0, 0...] encodée en plage et retourne le nombre
     de bits optimal pour encoder cette liste """
    max_bits = bits(max(L))  # Nombre de bits pour encoder le nombre maximal de la liste sur un bloc
    min_len_bits = None
    min_len = None
    for b in range(1, max_bits + 1):
        n = 0
        for val in L:
            v = val
            while v > 0:
                n += b
                v -= 2**b - 1
                if v > 0:
                    n += b
        if min_len is None or n < min_len:
            min_len = n
            min_len_bits = b
    return min_len_bits

# def calc_size(non_zero_val_matrix, non_zero_pos_matrix):
#     """ Retourne le nombre de bits nécessaire pour encoder l'image compressée
#     non_zero_val_matrix est une liste de liste comportant les coefficients non nuls de chaque bloc
#     non_zero_pos_matrix est une liste de liste comportant les positions des coefficients non nuls de chaque bloc,
#     encodée en plage """
#     non_zero_val_bits = []
#     for non_zero_val_list in non_zero_val_matrix.flatten():
#         if non_zero_val_list:
#             # 3 bits pour le nombre de bits par élement, ce qui permet un élement max de 2^(2^3) - 1 = 255 dans la
#             # matrice de DCT quantifiée
#             # 1 bit pour le signe et le nombre de bits nécessaire pour encoder la valeur maximale du bloc en valeur
#             # absolue
#             non_zero_val_bits.append(3 + len(non_zero_val_list) * (1 + bits(max(np.abs(np.array(non_zero_val_list))))))

#     non_zero_pos_bits = []
#     for non_zero_pos_list in non_zero_pos_matrix.flatten():
#         non_zero_pos_bits.append(len(non_zero_pos_list) * pos_encoding(np.abs(np.array(non_zero_pos_list))))

#     N = sum(non_zero_pos_bits) + sum(non_zero_val_bits)
#     return N


#Lecture de l'image
im_ori=cv2.imread(r"image4.png",0)

#extraction taille et poid
N=np.size(im_ori)
Height,Width=np.shape(im_ori)
P_or=N*8

print("p_or=", P_or)

#Transformée en ondelette
onde="haar" #'bior1.3'  'haar'
level=3

transf_wct = pywt.wavedec2(im_ori, onde, 'symmetric', level)

#affichage transformée
arraywvt,coeff2_slices=pywt.coeffs_to_array(transf_wct)
fig = plt.figure(figsize=(12, 3))
plt.imshow(arraywvt, interpolation="nearest", cmap=plt.cm.gray)

#affichage histogramme
fig2 = plt.figure(figsize=(12, 3))

for i in range (1,level+1):
    echelle = transf_wct[i][1]+transf_wct[i][1]+transf_wct[i][2]

    histograme,bins=np.histogram(echelle,np.arange(-100,100,1))
    ax=fig2.add_subplot(level,1,i)
    ax.set_title('échelle'+str(i))
    ax.bar(np.arange(-100,99,1),histograme)


#pas de quantification pour les différents niveaux
deltacoef=[1,2,10,100]

im_quantif=transf_wct

#quantification
im_quantif[0]=np.sign(im_quantif[0])*np.floor(im_quantif[0]/deltacoef[0])

for j in range(1,len(deltacoef)):

    newcoeff0=np.sign(im_quantif[j][0])*np.floor(np.abs(im_quantif[j][0])/deltacoef[j])
    newcoeff1=np.sign(im_quantif[j][1])*np.floor(np.abs(im_quantif[j][1])/deltacoef[j])
    newcoeff2=np.sign(im_quantif[j][2])*np.floor(np.abs(im_quantif[j][2])/deltacoef[j])
    
    im_quantif[j]=(newcoeff0,newcoeff1,newcoeff2)


#====> Je plante ici <=====

#codage en plage puis calcule taux compression
im_plage=[[],[[],[],[]],[[],[],[]],[[],[],[]]] #si on convertit en liste, on devrait avoir un truc du genre. 
#L'idée est qu'on a besoin d'une variable de ce genre pour reconstruire la transformée en ondelette plus tard


#tentative désespéré de faire un codage par plage, mais la fct est pas adaptée au codage en ondelette.
# On a des valeurs de int et pas des binaires (mais les variables sont des floats et je vois pas comment tout convertir efficacement)
for i in range(0,np.shape(im_quantif[0])[0]):
    im_plage[0].append(plage(im_quantif[0][i].tolist()))

for i in range(1,3):
    for j in range(0,2):
        for k in range(0,np.shape(im_quantif[i][j].tolist())[0]):
            im_plage[i][j].append(plage(im_quantif[i][j].tolist()[k]))


max_bits = bits(max(im_plage))
P_comp = N * max_bits
print("P_comp=",P_comp)
print("Taux de compression=",P_or/P_comp)



#décodage plage
im_unplage=un_plage(im_plage)

#reconstruction de la transformée en ondelette
r=0.5 #variable de décodage inverse, comprise dans [0;1]
im_dequantif=[]
im_dequantif[0] = (im_unplage[0] + r *np.sign(im_unplage[0]))*deltacoef[0]

for j in range(1,len(deltacoef)):
    newcoeff0=(im_unplage[j][0] + r *np.sign(im_unplage[j][0]))*deltacoef[j]
    newcoeff1=(im_unplage[j][1] + r *np.sign(im_unplage[j][1]))*deltacoef[j]
    newcoeff2=(im_unplage[j][2] + r *np.sign(im_unplage[j][2]))*deltacoef[j]
    
    im_dequantif[j]=(newcoeff0,newcoeff1,newcoeff2)


#reconstruction de l'image
im_decomp = pywt.waverec2(im_dequantif, onde, 'symmetric')



#métrique de qualité utilisé: RMSE = sqrt(||Icomp-Ior||^2 / N)
RMSE=np.sqrt(pow(np.linalg.norm(np.array(im_decomp)-np.array(im_ori)),2)/N)
print("RMSE=",RMSE)


#affichage image finale
plt.show()
while True:
    key = cv2.waitKey(1) #on évalue la touche pressée
    cv2.imshow('image_Originale', im_ori) #affichage
    cv2.imshow('image_Compressée', im_decomp.astype('uint8')) #affichage
    plt.show()
    if key & 0xFF == ord('q'): #si appui sur 'q'
        break #sortie de la boucle while


cv2.destroyAllWindows()