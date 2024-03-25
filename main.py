import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow_estimator import estimator
import os

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

dftrain, dfeval, y_train, y_eval = train_test_split(x_dataset, y_dataset, shuffle = False, test_size = 0.3) #split data up

#parse data into feature columns
features = ["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]
feature_columns = []
for feature_name in features:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype = tf.float32))
    

#get input functions
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

#create model
linear_est = estimator.LinearClassifier(feature_columns=feature_columns)

#train
linear_est.train((train_input_fn))

#check accuracy
#result = linear_est.evaluate(eval_input_fn)
os.system('clear')
#print(f"Accuracy: {result['accuracy']}")

prediction_data = list(linear_est.predict(eval_input_fn))
prediction = prediction_data.pop(0)

print(f"Predictions: {prediction}")

print(f"Actual Values: \n{y_eval}")
