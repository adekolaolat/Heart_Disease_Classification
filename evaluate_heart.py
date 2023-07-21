import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import  accuracy_score
from tabpy.tabpy_tools.client import Client


# load saved model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to evaluate model
def get_accuracy(model_path, test_file_path):

    # load test data into dataframe
    df = pd.read_csv(test_file_path)
    X_data= df.drop('target', axis=1)

    # Normalize features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_data)

    # Get labels
    y = df['target']

    # load model
    model = load_model(model_path)

    # predict with test set with model
    pred = model.predict(X)

    # get accuracy
    accuracy_value = accuracy_score(y, pred)

    accuracy = str(round(accuracy_value,3) * 100)
    return accuracy+'%'
    

client = Client('http://localhost:9004/')

# Register functions
client.deploy('getAccuracy', get_accuracy, 'Calculates the test accuracy', override=True)


 