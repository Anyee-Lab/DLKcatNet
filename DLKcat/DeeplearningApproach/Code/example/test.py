
import sys
import dlkcat_prediction
import prediction_for_input
import test_2
def dlkcat_pridict(a,b,c,d):
    print(torch.cuda.is_available())
    x = dlkcat_prediction.predict_for_one(a,b,c,d)
    return x

# def dlkcat_pridict_for_input():
#     x = prediction_for_input.predict_for_input()

# Predictor = test_2.DLKcatPrediction("bin/Debug/net9.0/python_embedded/python_code")

# def dlkcat_pridict(name,smiles,sequence):
#     x=Predictor.predict_for_one(name,smiles,sequence)
#     return x

# def dlkcat_pridict_for_input(name):
#     Predictor.predict_for_input(name)