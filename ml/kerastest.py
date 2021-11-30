# 평균 기온에 따른 배추가격 예측
# year    avgTemp    minTemp    maxTemp    rainFall    avgPrice
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
datasets = pd.read_csv( "pricedata.csv" )
# sns.pairplot( datasets[["avgTemp", "minTemp", "maxTemp", "rainFall", "avgPrice"]], \
              # diag_kind="kde" )
# plt.show()

# print( datasets.describe() )
stats = datasets.describe()
stats.pop( "avgPrice" )
stats.pop( "year" )
stats = stats.transpose()
# print( stats )
norm_datasets = ( datasets-stats["mean"] ) / stats["std"]
# print( norm_datasets )
norm_datasets["avgPrice"] = datasets.loc[:, "avgPrice"]
norm_datasets = norm_datasets.drop( ["year"], axis=1 )
# print( norm_datasets )

train = norm_datasets.sample( frac=0.8, random_state=1 )
test = norm_datasets.drop( train.index )
train_data = train[["avgTemp", "rainFall"]]
train_label = train["avgPrice"]
test_data = test[["avgTemp", "rainFall"]]
test_label = test["avgPrice"]

from tensorflow.keras import Sequential, layers, optimizers, callbacks
model = Sequential([
        layers.Dense( 64, activation="relu", input_shape=[2] ),
        layers.Dense( 64, activation="relu" ),
        layers.Dense( 1 )
    ])
optimizer = optimizers.RMSprop( 0.001 )
model.compile( optimizer=optimizer, loss="mse", metrics=["mae", "mse"] )
stop = callbacks.EarlyStopping( monitor="loss", patience=10 )
hist = model.fit( train_data, train_label, epochs=10, \
                  validation_split=0.2, callbacks=[stop] )
predict = model.predict( test_data ).flatten()
print( predict )

plt.scatter( test_label, predict, alpha=0.5 )
plt.xticks( [2000, 8000])
plt.yticks( [2000, 8000])
plt.xlabel( "Label" )
plt.ylabel( "predict" )
plt.show()








