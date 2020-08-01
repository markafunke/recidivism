import pickle
import pandas as pd
import numpy as np

# read in the model
my_model = pickle.load(open("my_pickled_model2.p","rb"))

# create a function to take in user-entered amounts and apply the model
def recidivism(amounts_float, model=my_model):
    
    # inputs into the model
    input_df = [amounts_float]

    # make a prediction
    prediction = my_model.predict(input_df)[0]

    # return a message
    message_array = ["Predict Recidivism",
                     "Predict No Recidivism"]

    return message_array[prediction]
