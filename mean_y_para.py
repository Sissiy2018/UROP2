# mlp for bimodal distribution
#%%
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization,Input,Model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# Define the means and standard deviations for the two Gaussian distributions
mean1_range = np.arange(10, 1001, 10)
mean2_range = np.arange(10, 1001, 10)
std_dev = 50
count = 0

# Initialize empty arrays to store the samples and parameters
samples = np.empty((10000, 500), dtype=np.float64)
para = np.empty((10000, 2), dtype=np.float64)

# Generate samples from each distribution
for mean1 in mean1_range:
    for mean2 in mean2_range:
        # Generate 250 samples from the first Gaussian distribution
        dist1_samples = np.random.normal(mean1, std_dev, size=250)
        # Generate 250 samples from the second Gaussian distribution
        dist2_samples = np.random.normal(mean2, std_dev, size=250)
        # Concatenate the samples from both distributions
        dist_samples = np.concatenate([dist1_samples, dist2_samples])
        #para_com = np.concatenate([mean1, mean2])
        # Append the samples to the main array
        samples[count] = dist_samples
        para[count] = np.array([mean1,mean2])
        count += 1

print(samples[1])
print(para[0])

def aleatoric_loss(y_true, y_pred):
    se = K.pow((y_true[:,:4]-y_pred[:,:4]),2)
    inv_std = K.exp(-y_pred[:,4:])
    mse = K.mean(K.batch_dot(inv_std,se))
    reg = K.mean(y_pred[:,4:])
    return 0.5*(mse + reg)

X = para
y = samples
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# determine the number of input and output features
n_features = X_train.shape[1]
input_shape = (n_features,) 
output_shape = (y_train.shape[1],)

neurons = 100
layers = 3
dropout_rate = 0.2
epochs = 500
## define model
model = Sequential()
model.add(Dense(100, activation='relu', kernel_initializer='uniform', input_shape=(n_features,))) # special for only one dimension
for i in range(layers):
    model.add(Dropout(rate = dropout_rate))
    model.add(Dense(neurons, kernel_initializer='uniform', activation='relu'))
model.add(Dense(output_shape[0]))
opt = tf.keras.optimizers.Adam(learning_rate=0.001,
                               beta_1=0.9,beta_2=0.999,epsilon=1e-09,)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
print(model.summary())
history = model.fit(X_train, y_train, batch_size=int(len(X_train)/3), epochs = epochs, shuffle=True, validation_data=(X_test, y_test), use_multiprocessing=True)

train_mse = model.evaluate(X_train, y_train, verbose=0)
test_mse = model.evaluate(X_test, y_test, verbose=0)
# plot loss during training
plt.figure(1)
plt.title('Loss / Mean Squared Error')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

y_pred =  model.predict(X_test)
h = [0,1,2,3]
y_val = y_test
for i in range(2):
    plt.figure(i+2)
    plt.plot(y_val[:,i],y_val[:,i],'r.')
    plt.plot(y_val[:,i],y_pred[:,i],'ko',alpha=0.4)
    plt.figure(7)
    y = abs(y_pred[:,i] - y_val[:,i])/np.max(abs(y_pred[:,i] - y_val[:,i]))
    plt.plot(y_val[:,i]/np.max(y_val[:,i]),y,'o')
plt.show()

model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='uniform', input_shape=(n_features,))) # special for only one dimension
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(8, activation='relu', kernel_initializer='uniform'))
model.add(Dense(output_shape[0]))
# compile the model
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
# configure early stopping
es = EarlyStopping(monitor='val_loss', patience=5)
# summarize the model
model.summary()
#plot_model(model, 'model.png', show_shapes=True)
# Save the weights using the `checkpoint_path` format
#model.save_weights(checkpoint_path.format(epoch=0))
# fit the model
model.fit(X_train, y_train, batch_size=int(len(X_train)/3), epochs = 10, shuffle=True,validation_data=(X_test, y_test))

## define model
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='uniform', input_shape=(n_features,))) # special for only one dimension
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(8, activation='relu', kernel_initializer='uniform'))
model.add(Dense(output_shape[0]))
# compile the model
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
# configure early stopping
es = EarlyStopping(monitor='val_loss', patience=5)
# summarize the model
model.summary()
#plot_model(model, 'model.png', show_shapes=True)
# Save the weights using the `checkpoint_path` format
#model.save_weights(checkpoint_path.format(epoch=0))
# fit the model
model.fit(X_train, y_train, batch_size=int(len(X_train)/3), epochs = 10, shuffle=True,validation_data=(X_test, y_test))


# load the dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
df = read_csv(path, header=None)
print(df.head())
# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
print(X.shape)
print(y.shape)
# ensure all data are floating point values
X = X.astype('float32')
# encode strings to integer
y = LabelEncoder().fit_transform(y)
print(y)
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# determine the number of input features
n_features = X_train.shape[1]
# define model
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,))) # special for only one dimension
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# configure early stopping
es = EarlyStopping(monitor='val_loss', patience=5)
# summarize the model
model.summary()
plot_model(model, 'model.png', show_shapes=True)
# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))
# fit the model
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0,callbacks=[cp_callback])

latest = tf.train.latest_checkpoint(checkpoint_dir)
# Create a new model instance
model = create_model()
# Load the previously saved weights
model.load_weights(latest)

# save model to file
model.save('model.h5')
# load the model from file
model = load_model('model.h5')
# evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)
# make a prediction
row = [5.1,3.5,1.4,0.2]
yhat = model.predict([row])
print('Predicted: %s (class=%d)' % (yhat, np.argmax(yhat)))
# %%
