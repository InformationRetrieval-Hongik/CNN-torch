from keras.layers import Embedding, Dense, LSTM, Flatten, Conv2D, MaxPooling2D, Reshape
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from DataSet import loadDocs

# ==================================================================================================
# ========================= generate simple test model and test data set. ==========================
# ==================================================================================================
train_x = loadDocs("./Data/train_x.pkl")
train_y = loadDocs("./Data/train_y.pkl")
test_x = loadDocs("./Data/test_x.pkl")
test_y = loadDocs("./Data/test_y.pkl")
lookupTable = loadDocs("./Data/lookupTable.pkl")

model = Sequential()
vocab_size = len(lookupTable.keys())
embedding_dim = 128
maxLen = train_x.shape[1]
print(train_x.shape)
print(vocab_size, maxLen)

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=maxLen))
print(model.output_shape)
# model.add(Dropout(0.3))
model.add(Reshape((embedding_dim, maxLen, -1)))
print(model.output_shape)
model.add(Conv2D(50, (5, 80), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 1)))
print(model.output_shape)

model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# # earlyStop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=4)
# modelCheck = ModelCheckpoint("1D_cnn_best_model.h5", monitor="val_acc", mode="max", verbose=1, save_best_only=True)

# model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
# history = model.fit(train_x, train_y, epochs=15, callbacks=[modelCheck], batch_size=60, validation_split=0.2)

# loaded_model = load_model("1D_cnn_best_model.h5")
# print("\ntest accuracy : %.4f" % (loaded_model.evaluate(test_x, test_y)[1]))
