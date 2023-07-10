# mlp for bimodal distribution
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
import tensorflow as tf

# Define the means and standard deviations for the two Gaussian distributions
mean1_range = np.arange(10, 1001, 10)
mean2_range = np.arange(10, 1001, 10)
std_dev_range = np.arange(10, 101, 10)
count = 0

# Initialize empty arrays to store the samples and parameters
samples = np.empty((100000, 500), dtype=np.float64)
para = np.empty((100000, 3), dtype=np.float64)

# Generate samples from each distribution
for mean1 in mean1_range:
    for mean2 in mean2_range:
        for std_dev in std_dev_range:
            # Generate 250 samples from the first Gaussian distribution
            dist1_samples = np.random.normal(mean1, std_dev, size=250)
            # Generate 250 samples from the second Gaussian distribution
            dist2_samples = np.random.normal(mean2, std_dev, size=250)
            # Concatenate the samples from both distributions
            dist_samples = np.concatenate([dist1_samples, dist2_samples])
            # Append the samples to the main array
            samples[count] = dist_samples
            para[count] = np.array([mean1, mean2,std_dev])
            count += 1

print(samples[1])
print(para[0])

X = samples
y = para
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# determine the number of input and output features
n_features = X_train.shape[1]
input_shape = (n_features,) 
output_shape = (y_train.shape[1],)

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