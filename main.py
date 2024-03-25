import pandas as pd
import tensorflow as tf
import tensorflow._api.v2.compat.v2.feature_column
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

x_dataset = pd.read_csv("Cancer_Data.csv")
x_dataset.pop("id") #get rid of id number
x_dataset.dropna(axis=0, how="any")
y_dataset = (x_dataset.pop("diagnosis")).replace({'B':0, 'M':1}) #convert to numerical data

#Make input function
def make_input_fn(data_df, label_df, num_epochs = 10, shuffle=True, batch_size = 12):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))

        if shuffle:
            ds = ds.shuffle(1000)

        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

dftrain, dfeval, y_train, y_eval = train_test_split(x_dataset, y_dataset, shuffle = False, test_size = 0.3)