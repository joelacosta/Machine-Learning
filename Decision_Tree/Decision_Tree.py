#!/usr/bin/env python
# coding: utf-8

# In[31]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np
from sklearn.datasets import make_blobs


# In[255]:


#Otra vez uso make_blobs para empezar a entender los modelos. Spliteo los datos en los que se van a usar para entrenar y para testear
X,y=make_blobs(n_samples=150,centers=2,n_features=2,cluster_std=3.3)
X,X_test,y,y_test=train_test_split(X,y)
#Los datos tiene 2 features, las coordenadas (x,y) y un target que puede ser clase 1 o 2
plt.figure(figsize=(10,8));
plt.scatter(X[:,0],X[:,1],c=y,marker='o',edgecolor='k',label='Train',zorder=3);


# In[253]:


#Asi se define el arbol y de esta forma se puede visualizar el mapa del modelo que queda
arbol = DecisionTreeClassifier(max_depth=2,random_state=0);
arbol.fit(X,y);
tree.plot_tree(arbol)
plt.show()


# In[247]:


def arbol_decision(m):
    plt.scatter(X[:,0],X[:,1],c=y,marker='o',edgecolor='k',label='Train',zorder=3);
    arbol = DecisionTreeClassifier(max_depth=m,random_state=0);
    arbol.fit(X,y);
    feature=arbol.tree_.feature;  #Esto es para extraer información de cada nodo del arbol
    left=arbol.tree_.children_left;
    threshold=arbol.tree_.threshold;
    right=arbol.tree_.children_right;
    eps=1.5
    x_min, x_max=min(X[:,0])-eps, max(X[:,0])+eps
    y_min, y_max=min(X[:,1])-eps, max(X[:,1])+eps
    x_plot=np.arange(x_min,x_max,0.2)
    y_plot=np.arange(y_min, y_max,0.2)
    mask_1=np.zeros((len(y_plot),len(x_plot)))
    plt.xlim([x_min,x_max])
    plt.ylim([y_min,y_max])
    plt.xticks([]);
    plt.yticks([]);
    #Creamos una mascara para ver que es lo que dice el modelo sobre el espacio (x,y)
    for i in range(len(x_plot)):
        for j in range(len(y_plot)):
            mask_1[j,i]=arbol.predict([[x_plot[i],y_plot[j]]])
    plt.contourf(x_plot,y_plot,mask_1,extend='both',alpha=0.8,zorder=1);
    #Todo lo que vien acá es para graficar las lineas que separan las regions para cada orden del modelo
    limites={0:[x_min,x_max,y_min, y_max]}
    n=[0]
    for i in n:
        x_1,x_2,y_1,y_2=limites[i]
        if feature[i]==1:
            plt.hlines(threshold[i],x_1,x_2,color='k',linestyles='--',linewidth=2)
            if left[i]!=-1:
                n.append(left[i])
                limites[left[i]]=[x_1,x_2,y_1,threshold[i]]
                n.append(right[i])
                limites[right[i]]=[x_1,x_2,threshold[i],y_2]
        if feature[i]==0:
            plt.vlines(threshold[i],y_1,y_2,color='k',linestyles='--',linewidth=2)
            if left[i]!=-1:
                n.append(left[i])
                limites[left[i]]=[x_1,threshold[i],y_1,y_2]
                n.append(right[i])
                limites[right[i]]=[threshold[i],x_2,y_1,y_2]
    return '%Train: {:.3f}'.format(arbol.score(X,y))+ '. %Test: {:.2f}'.format(arbol.score(X_test,y_test))


# In[248]:


#Por ejemplo, se puede ver todo junto corriendo simplemente esto
plt.figure(figsize=(10,8));
arbol_decision(1)


# In[258]:


plt.figure(figsize=(16,8))
ax1 = plt.subplot(221)
score=arbol_decision(1)
plt.title('Depth=1. '+score)


ax2= plt.subplot(222)
score=arbol_decision(2)
plt.title('Depth=2. '+score)

ax3=plt.subplot(223)
score=arbol_decision(3)
plt.title('Depth=3. '+score)

ax3=plt.subplot(224)
score=arbol_decision(4)
plt.title('Depth=4. '+score)

plt.savefig('tree.png',dpi=150)


# In[ ]:




