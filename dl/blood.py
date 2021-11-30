# 체중과 나이로 혈당 예측
import numpy as np
dataset = np.loadtxt("blood.csv",delimiter=",", dtype=np.float32)
# print(dataset.shape)
# print(dataset)
x_data = np.array( dataset[:, 1])       # 체중
y_data = np.array( dataset[:, 2])       # 나이
z_data = np.array( dataset[:, 3])       # 혈당
#print(z_data)
#print(z_data.shape)         # (130, )
import matplotlib.pyplot as plt
# fig = plt.figure( figsize = (5, 5))
# ax = fig.add_subplot(1, 1, 1, projection="3d")
# ax.scatter(x_data, y_data, z_data)
# ax.set_xlabel("weight")
# ax.set_ylabel("age")
# ax.set_zlabel("blood")
# plt.show()

train = np.array( dataset[:, 1:3], dtype=np.float32)    # 독립변수        몸무게 나이
label = np.array( dataset[:, 3], dtype=np.float32)      # 종속변수         혈당치

from tensorflow import keras 
rmsprop = keras.optimizers.RMSprop( lr = 0.01 )
model = keras.Sequential()
model.add(keras.layers.Dense(1, input_shape=(2, )))
model.compile( loss="mse" , optimizer = rmsprop )
#print(model.summary())

stop = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 5)
hist = model.fit(train, label, epochs=250, validation_split=0.2, callbacks=[stop])

# print(hist.history.keys())
# plt.plot(hist.history["loss"])
# plt.xlabel("epochs")
# plt.ylabel("loss")
# plt.show()

W, b = model.get_weights()
print(W,b)

x= np.linspace(20, 100, 50 ).reshape(50,1)  # 체중
y = np.linspace(10, 70, 50).reshape(50, 1) # 나이
test = np.concatenate((x,y), axis=1)
z = np.matmul( test, W) + b

fig = plt.figure(figsize = (5,5 ))
ax = fig.add_subplot(1, 1, 1, projection="3d")
ax.scatter(x, y , z)
ax.scatter(x_data, y_data, z_data)
ax.set_xlabel("weight")
ax.set_ylabel("age")
ax.set_zlabel("blood")
plt.show()
