#!/usr/bin/env python
# coding: utf-8


#importo las dependencias que voy a usar
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



#Levanto la info de la base de datos. Esta base de datos viene con 70k de numeros escritos a mano separados en 60K que voy a usar para entrenar y
#otros 10K que se pueden usar para testear. Las imagenes estan normalizdas en 0 y 255. Las quiero normalizadas en 0 y 1
mnist = tf.keras.datasets.mnist 

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

np.shape(x_train)


#Defino el modelo. Se pueden poner varias capas, cada una con sus parametros para entrenar. 
model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), #Esta simplemente pone todos los datos de la imagen en una sola tira.
    #tf.keras.layers.Dropout(0.2),                 #Esta se usa cuando se entrena y pone 0.2% de los datos a 0. Sirve para no sobrefitear cosas
    tf.keras.layers.Dense(10)                      #La ultima siempre es del output. Aca le estoy diciendo que al final tiene que devolver 10 valores (luego se interpetan estos valores como la probabilidad de que cada imagen corresponda a cierto numero)
])                                                 #Aca se puede poner tantas capas como se quiera. Dense simplemente hace una Transformación matricial Y=A*X+B de los parametros.
model.summary()                                    # Si miramos tenemos 7850 parametros que son las 7840 componentes de la matriz A + las 10 componentes del vector B


prediccion = model(x_train[:1]).numpy() #Asi se consiguen las predicciones del modelos. En este caso como no esta entrenado los parametros silamente estan inicializados y la predicción es aleatoria.
antes=tf.nn.softmax(prediccion).numpy() #Esto devuelve una suerte de logaritmo normalizado de las probabilidades. Para invertirlo y conseguir las probabilidades se usa la función softmax
antes


#Hay que definir que función va a querer minimizar el modelo mientras se va entrenando. Esto es lo que define en ultima instancia como se van a interpretar las salidas del modelo.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) #Ver la documentación para esto. Pero al minimizar esta función lo que se hace es lo que mejore las predicciónes 
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])   #Aca se compila el modelo, se elige el algoritmo que se usa para ir ajustando los parámetros


#Con este comando se entrena. Los y_train son simplemente enteros para cada imagen que dice que numero está en la imagen. Que se tenga que pasar un enetero para cada imagen y no una tira de probabilidades se debe a la función de perdida que se eligio y a la metrica
model.fit(x_train, y_train, epochs=5)


#Si queremos ver las % de que la cuarta imagen sea cada numero evaluamos el modelo.
prob=tf.nn.softmax(model([x_test[4:5]])).numpy()
prob


numeros=[0,1,2,3,4,5,6,7,8,9]
despues=prob



f, (ax1,ax2)=plt.subplots(nrows=1,ncols=2,width_ratios=(1,3),figsize=(12,8))
ax1.imshow(x_test[4],cmap='gray')
ax1.axis('Off')
ax1.set_xlabel(r'Imagen')
ax2.set_xticks(numeros)
ax2.grid(linestyle='dashed')
ax2.bar(numeros,antes,width=0.5,align='edge',label='Sin entrenar')
ax2.bar(numeros,despues,width=-0.5,align='edge',label='Entrenada')
ax2.legend()
ax2.set_ylim([0,1])
ax2.set_ylabel('Probabilidades')
plt.savefig('figura.png',dpi=300)

