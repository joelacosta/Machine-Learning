#!/usr/bin/env python
# coding: utf-8

# In[21]:


import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[253]:


#Levanto la base de datos. Se puede ver las entradas del diccionario con housing.keys()
housing = fetch_california_housing()
print(housing['DESCR'])


# In[254]:


#Spliteamos la data 
X_train,X_test,y_train,y_test=train_test_split(housing['data'],housing['target'])
#Estoy ajusta un hiperplano en el espacio de AtributesXtarget. Es simplemente un ajuste lineal de los datos. Minimiza las ditacias cuadraticas.
lr = LinearRegression().fit(X_train, y_train)
#Las pendientes y ordenada se pueden ver asi
print(lr.coef_)
print(lr.intercept_)
print('Eficiencia en el training set: {:.2f}'.format(lr.score(X_train,y_train)))
print('Eficiencia en el test set: {:.2f}'.format(lr.score(X_test,y_test)))


# In[57]:


#Este modelo ajusta cuadrados minimos con el constraint de que la suma de coeficientes al cuadrado sea minima. El alpha controla que tan importante es eso 
#ver https://scikit-learn.org/stable/modules/linear_model.html
#Se juega con el alpha para que esto ajuste mejos
from sklearn.linear_model import Ridge
ridge=Ridge(alpha=10000).fit(X_train,y_train)
print('Eficiencia en el training set: {:.2f}'.format(ridge.score(X_train,y_train)))
print('Eficiencia en el test set: {:.2f}'.format(ridge.score(X_test,y_test)))


# In[82]:


#Laso es similar al anterior pero minimiza una funcion que prefiere que la mayor cantidad posible de coeficientes sean 0.
#Es util para reducir el espacio de features a las minimas posibles
from sklearn.linear_model import Lasso
lasso=Lasso(alpha=1).fit(X_train,y_train)
print('Eficiencia en el training set: {:.2f}'.format(lasso.score(X_train,y_train)))
print('Eficiencia en el test set: {:.2f}'.format(lasso.score(X_test,y_test)))
print(lasso.coef_)
print("Numero de  features: {}".format(np.sum(lasso.coef_ != 0)))


# In[281]:


#Los modelos anteriores tenian un target continuo. Para en caso de selectores tambien se puede hacer un modelo lineal. El hiperplano que arma en el espacio de features "separa" las regiones
#Lo podemos comparar con el modelo de primeros vecinos. Todo esto esta en el codigo de knn
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
X,y=make_blobs(n_samples=150,centers=2,n_features=2,cluster_std=2.5)
X,X_test,y,y_test=train_test_split(X,y)

knn_1=KNeighborsClassifier(n_neighbors=1).fit(X,y)
log_reg=LogisticRegression().fit(X,y)
log_reg100=LogisticRegression(C=100).fit(X,y)
modelos=[knn_1,log_reg,log_reg100]
ef_train=[str(round(i.score(X,y),2)) for i in modelos]
ef_test=[str(round(i.score(X_test,y_test),2)) for i in modelos]


# In[282]:


eps=1.5
x_min, x_max=min(X[:,0])-eps, max(X[:,0])+eps
y_min, y_max=min(X[:,1])-eps, max(X[:,1])+eps
x_plot=np.arange(x_min,x_max,0.2)
y_plot=np.arange(y_min, y_max,0.2)
mask_1=np.zeros((len(y_plot),len(x_plot)))
for i in range(len(x_plot)):
    for j in range(len(y_plot)):
        mask_1[j,i]=knn_1.predict([[x_plot[i],y_plot[j]]])


# In[283]:


#Aca se ve la estructura de la data y como se separa por clases
plt.figure(figsize=(10,8));
plt.contourf(x_plot,y_plot,mask_1,extend='both',alpha=0.8);
plt.scatter(X[:,0],X[:,1],c=y,marker='o',edgecolor='k',label='Train',s=60);
plt.scatter(X_test[:,0],X_test[:,1],c=y_test,marker='^',edgecolor='k',label='Test',s=60);
plt.xlim([x_min+0.5,x_max-0.5]);
plt.ylim([y_min+0.5,y_max-0.5]);

m=-log_reg.coef_[0][0]/log_reg.coef_[0][1]
b=-log_reg.intercept_/log_reg.coef_[0][1]
b100=-log_reg100.intercept_/log_reg100.coef_[0][1]
m100=-log_reg100.coef_[0][0]/log_reg100.coef_[0][1]

plt.plot(x_plot,m*x_plot+b,label=r'LogRegressor: C=1. %Train: '+ef_train[1]+'. %Test: '+ef_test[1],c='r')
plt.plot(x_plot,m100*x_plot+b100,label=r'LogRegressor: C=100. %Train: '+ef_train[2]+'. %Test: '+ef_test[2],c='k',linestyle='--')
eps=1
plt.legend()
plt.savefig('comparacion.png',dpi=300)


# In[285]:


#Es posible ver que features son las mas relevantes. Usando los datos de cancer podemos jugar con el C del modelos
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer['data'], cancer['target'])

logreg = LogisticRegression(C=1,max_iter=1000).fit(X_train, y_train);
logreg100 = LogisticRegression(C=1000,max_iter=1000).fit(X_train, y_train);
logreg001 = LogisticRegression(C=0.001,max_iter=1000).fit(X_train, y_train);


# In[286]:


#Y plotear los coeficientes relativos de cada uno de los modelos para cada feature

plt.figure(figsize=(10,8));
etiquetas=cancer['feature_names']
modelos=[logreg001, logreg, logreg100]
coeficientes=logreg.coef_[0]
coeficientes100=logreg100.coef_[0]
coeficientes001=logreg001.coef_[0]
ef_train=[str(round(i.score(X_train,y_train),2)) for i in modelos]
ef_test=[str(round(i.score(X_test,y_test),2)) for i in modelos]
plt.scatter(range(len(etiquetas)),coeficientes001/max(abs(coeficientes001)),label='C=001. %Train='+ef_train[0]+'. %Test='+ef_test[0],marker='x',s=45);
plt.scatter(range(len(etiquetas)),coeficientes/max(abs(coeficientes)),label='C=1. %Train='+ef_train[1]+'. %Test='+ef_test[1],s=45);
plt.scatter(range(len(etiquetas)),coeficientes100/max(abs(coeficientes100)),label='C=1000. %Train='+ef_train[2]+'. %Test='+ef_test[2],marker='^',s=45);
plt.xticks(range(len(etiquetas)), etiquetas,rotation=90);
plt.yticks(np.arange(-1,1.5,0.25))
plt.grid(linestyle='--',color='k',alpha=0.65)
plt.ylabel('Peso relativo')
plt.ylim([-1.25,1.25])
plt.legend();
plt.savefig('coeficientes_relativos.png',dpi=300)


# In[ ]:




