#!/usr/bin/env python
# coding: utf-8

import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


iris_dataset=load_iris() #Esta es una base de datos muy conocida de medicion de petalos de iris para clasificarlas
iris_dataset.keys() #Esta almacenada como un diccionario

print(iris_dataset['DESCR'])


iris_dataset['data'].shape , type(iris_dataset['data']) #La data es un numpy array de 150 filas (distintas mediciones) y 4 columnas (las 4 features de las plantas)



X_train,X_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target']) #Esto randomiza y splitea los datos para entrentar 3/4 pasan a train y el resto se reserva para testear.
X_train.shape, type(X_train)



#Para entender más la data que hay guardada, se puede ver la correlación para cada par de features. Esto se hace muy simple con pandas.
iris_dataframe=pd.DataFrame(X_train,columns=iris_dataset['feature_names']);
pd.plotting.scatter_matrix(iris_dataframe,c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=.8);



#Como la data esta bastante agrupada, un modelo apropiado aca para predecir seria ver cuales son los puntos mas cercados del nuevo dato y asignarle simplemente esa clase.
#Esto se puede harcodear simplemente minimizando la distancia

def prediccion(X_train,y_train,X_test):
    v=[]
    for i in X_test:
        dist=[np.dot(i-j,i-j) for j in X_train]
        v.append(y_train[np.where(dist==min(dist))[0][0]])
    return v
    
pred_hrc=prediccion(X_train,y_train,X_test)
print(pred_hrc)

#Este forma de predecir la clase nueva tiene una eficiencia de 
np.mean(y_test==pred_hrc)


#En machine learning este tipo de modelos se llama de k-neighbors. En donde lo que yo hice es la primera aproximación y el modelo no tiene parametros para entrenar.
#Se puede hacer que la metrica sea distinta, que pesen mas algunas direcciones o considerar mayor cantidad de vecinos.
#Todo esto define el modelo de vecinos que se va a usar. Por ejemplo el modelo a primeros vecinos se escribe asi:
knn=KNeighborsClassifier(n_neighbors=1)   #Este es el unico parametro que importa en este caso
knn.fit(X_train, y_train)                 #Le damos los datos para que entrene (recuerde) 


X_new=np.array([[5.1,2.8,1.1,0.8]])  #Para predecir la clase de un nuevo dato, primero hay que revisar que el input sea un array de 1 fila y fetures columnas
X_new.shape

knn.predict(X_new)     #Y con este comando se predice.

pred=knn.predict(X_test), #Si quremos predecir los valores para X_test se usa el mismo comando
pred

np.mean(y_test==pred)  #Efectivamente este modelo es lo mismo que el que hardcodie antes.

#Para visualizar mejor el modelo de vecinos se puede constnstuir un data set artificial con la herramienta
from sklearn.datasets import make_blobs
X,y=make_blobs(n_samples=150,centers=2,n_features=2,cluster_std=2)
X,X_test,y,y_test=train_test_split(X,y)

#Aca se ve la estructura de la data y como se separa por clases
plt.figure(figsize=(10,8));
plt.scatter(X[:,0],X[:,1],c=y,marker='o',edgecolor='k',label='Train');

#Defino dos modelos disitintos. Uno que considera solamente a primeros vecinos y el otro a tercero vecinos
knn_1=KNeighborsClassifier(n_neighbors=1)
knn_1.fit(X,y)
knn_3=KNeighborsClassifier(n_neighbors=3)
knn_3.fit(X,y)

#Para graficar las regiones en las cuales el modelo predice la clase defino la grilla. Para toda la grilla veo y guarlo que es lo que predice el modelo
eps=1.5
x_min, x_max=min(X[:,0])-eps, max(X[:,1])+eps
y_min, y_max=min(X[:,1])-eps, max(X[:,1])+eps
x_plot=np.arange(x_min,x_max,0.2)
y_plot=np.arange(y_min, y_max,0.2)
mask_1=np.zeros((len(y_plot),len(x_plot)))
mask_3=np.zeros((len(y_plot),len(x_plot)))

for i in range(len(x_plot)):
    for j in range(len(y_plot)):
        mask_1[j,i]=knn_1.predict([[x_plot[i],y_plot[j]]])
        mask_3[j,i]=knn_3.predict([[x_plot[i],y_plot[j]]])


#Ploteo lo anterior
plt.figure(figsize=(10,8));
plt.contourf(x_plot,y_plot,mask_1,extend='both',alpha=0.8);
plt.scatter(X[:,0],X[:,1],c=y,marker='o',edgecolor='k',label='Train',s=60);
plt.scatter(X_test[:,0],X_test[:,1],c=y_test,marker='^',edgecolor='k',label='Test',s=60);
plt.xlim([x_min+0.5,x_max-0.5]);
plt.ylim([y_min+0.5,y_max-0.5]);
plt.legend()
plt.title('Modelo a primer vecino. {:.2f} de eficiencia'.format(knn_1.score(X_test,y_test)));
plt.savefig('knn_clasificador_1.png')


plt.figure(figsize=(10,8));
plt.contourf(x_plot,y_plot,mask_3,extend='both',alpha=0.8);
plt.scatter(X[:,0],X[:,1],c=y,marker='o',edgecolor='k',label='Train',s=60);
plt.scatter(X_test[:,0],X_test[:,1],c=y_test,marker='^',edgecolor='k',label='Test',s=60);
plt.xlim([x_min+0.5,x_max-0.5]);
plt.ylim([y_min+0.5,y_max-0.5]);
plt.legend()
plt.title('Modelo a terceros vecinos.{:.2f} de eficiencia'.format(knn_3.score(X_test,y_test)));
plt.savefig('knn_clasificador_3.png')

#K_nn regresor:
#Lo anterior se llama clasificador porque clasifica la nueva medicion en un conjunto discreto de clases. Cuando el espacio target es un continuo se lo llama regresor.
#Por ejemplo, generamos datos con ruido
n=60
X=10*np.random.random(n)
y=np.sin(2*X)+ X + np.random.normal(loc=0,scale=1.7,size=len(X));
plt.plot(X,y,'o');
X,X_test,y,y_test=train_test_split(X.reshape(-1,1),y);


#Definimos los regresores a primeros y 5tos vecinos
kn_reg=KNeighborsRegressor(n_neighbors=1);
kn_reg.fit(X,y);
kn_reg_3=KNeighborsRegressor(n_neighbors=5);
kn_reg_3.fit(X,y);


#Ploteo los datos relevantes
x_plot=np.linspace(0,10,300).reshape(-1,1)
plt.figure(figsize=(10,8))
plt.plot(x_plot,kn_reg.predict(x_plot),label='Regresor n=1. Score={:.2f}'.format(kn_reg.score(X_test,y_test)),c='k',alpha=0.8,zorder=1);
plt.plot(x_plot,kn_reg_3.predict(x_plot),label='Regresor n=5. Score={:.2f}'.format(kn_reg_3.score(X_test,y_test)),c='r',alpha=0.8,zorder=1);
plt.scatter(X,y,marker='o',label='Train',s=60);
plt.scatter(X_test,y_test,marker='^',label='Test',s=60);
plt.legend(loc=2);
plt.grid(linestyle='--',c='k',alpha=0.7)
plt.savefig('knn_regresor.png')

