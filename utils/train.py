import argparse

def getArgs():

    p = argparse.ArgumentParser()
    p.add_argument('training_path', type=str, help="Path to the training data in csv format")
    p.add_argument('--MAX_DF', type=int, help="Max number of data points from the training data")
    p.add_argument('--MAX_TOXIC', type=int, help="Max number of toxic data points")
    p.add_argument('model_put', type=str, help="Output path for the trained model")
    p.add_argument('model_type', type=int, help="1: svm\n2:RNN")

def train():

    args = getArgs()
    training_data = 

if __name__ == '__main__':
    trained_model = train()
