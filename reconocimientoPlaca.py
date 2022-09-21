import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm



def get_hog():
    winSize = (20,20)
    blockSize = (8,8)
    blockStride = (4,4)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 2
    histogramType = 0
    L2HysThreshold = 0.2 #Umbral por el que se multiplica la magnitud
    gammaCorrection = 1
    nlavels = 64
    signedGradient = True
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize,
                            nbins,derivAperture,winSigma,histogramType,
                            L2HysThreshold, gammaCorrection, nlavels, signedGradient)
    return hog

def escalar(img,m,n):
    if m > n:
        imgNueva = np.uint8(255*np.ones((m,round((m-n)/2),3)))
        escalada = np.concatenate((np.concatenate((imgNueva,img),
                                            axis=1),imgNueva),axis=1)
    else:
        imgNueva = np.uint8(255*np.ones((round((m-n)/2),n,3)))
        escalada = np.concatenate((np.concatenate((imgNueva,img),
                                          axis=0),imgNueva),axis=0)
    img = cv2.resize(escalada,(20,20))
    return img



def obtenerDatos():
    posiblesClases = ['0','1','2','3','4','5','6','7','8','9',
                  'A','B','C','D','E','F','G','H','J','K',
                  'L','M','N','P','Q','R','S','T','U','V',
                  'W','X','Y','Z']
    datos = []
    clases = []
    for i in range(1,26):
        for j in posiblesClases:
            img = cv2.imread(j+'-'+str(i)+".jpg")
            if img is not None:
                m,n,_ = img.shape
                if m!=20 or n!=20:
                    img = escalar(img,m,_)
                hog = get_hog()
                datos.append(np.array(hog.compute(img)))
                clases.append(np.where(np.array(posiblesClases)==j)[0][0])
    datos = np.array(datos)
    clases = np.array(clases)
    return datos, clases


def clasificadorCaracteres():
    datos, clases = obtenerDatos()
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(datos, clases)
    SVM = svm.SVC(kernel = 'linear', probability = True, 
                  random_state = 0, gamma = 'auto')
    SVM.fit(datos, clases)
    return knn, SVM




"""datos, clases = obtenerDatos()
# Evaluar el algoritmo KNN utilizando el metodo de validacion Hold Out
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(datos,clases,
                                        test_size=0.2, random_state=np.random)
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, Y_train)
rendimientoEntrenamientoKnn = (knn.score(X_train, Y_train)) * 100
print("Rendimiento de entrenamiento: " 
      + str(round(rendimientoEntrenamientoKnn,2))+"%")
rendimientoPruebaKnn = (knn.score(X_test, Y_test)) * 100
print("Rendimiento de entrenamiento: " 
      + str(round(rendimientoPruebaKnn,2))+"%")"""





