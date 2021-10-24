import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.metrics import mean_squared_error as RMSE
from sklearn.feature_selection import mutual_info_classif

input_train = {
    "y1": [1, 1, 0, 1, 2, 1, 2, 0],
    "y2": [1, 1, 2, 2, 0, 1, 0, 2],
    "y3": [0, 5, 4, 3, 7, 1, 2, 9],
    "output": [1, 3, 2, 0, 6, 4, 5, 7]
}

input_test = {
    "y1": [2, 1],
    "y2": [0, 2],
    "y3": [0, 1],
    "output": [2, 4]
}

train_df = pd.DataFrame.from_dict(input_train)
test_df = pd.DataFrame.from_dict(input_test)


def basis_func(j, x):
    return pow(np.linalg.norm(x), j)


temp_matrix = []

for i in range(8):
    new_x = []
    new_line = []

    new_x.append(input_train["y1"][i])
    new_x.append(input_train["y2"][i])
    new_x.append(input_train["y3"][i])

    for j in range(4):
        new_line.append(basis_func(j, new_x))

    print("Run #", i, "\nNew X: ", new_x, "\nNew Line: ", new_line)

    temp_matrix.append(new_line)

phi = np.matrix(temp_matrix)

test_temp = []

for output in input_train["output"]:
    test_temp.append([output])

test_outputs = np.matrix(test_temp)

phi_transposed = phi.transpose()
phi_pinv = np.linalg.pinv(phi)
weights = phi_pinv * test_outputs

weights_list = [weight[0] for weight in weights.tolist()]


def regression_model(x):
    res = 0.0
    for j in range(4):
        res += (weights_list[j] * basis_func(j, x))

    return res


print("\nPretty phi matrix:\n")
print(DataFrame(phi))

print("\nPretty transposed phi matrix:\n")
print(DataFrame(phi_transposed))

print("\nPretty pseudo-inverse phi matrix:\n")
print(DataFrame(phi_pinv))

print("\nPretty test outputs matrix:\n")
print(DataFrame(test_outputs))

print("\nPretty weights matrix:\n")
print(DataFrame(weights))


test_true = input_test["output"]
test_pred = []

for i in range(2):
    new_x = []

    new_x.append(input_test["y1"][i])
    new_x.append(input_test["y2"][i])
    new_x.append(input_test["y3"][i])

    result = regression_model(new_x)

    test_pred.append(result)

    print("Test Run #", i, "\nNew X: ", new_x, "\nNew Result: ",
          result, "\nActual Value: ", input_test["output"][i])

rmse = RMSE(test_true, test_pred, squared=False)

print("\nThe RMSE is ", rmse)


bin_out=list()
for el in input_train["output"]:
    if el >= 4:
        bin_out.append(1)
    else:
        bin_out.append(0)

bin_y3=[0,1,1,0,1,0,0,1]

#input_y1 = np.reshape([input_train["y1"],input_train["y2"],input_train["y3"]],(8,3))

#input = [input_train["y1"],input_train["y2"],input_train["y3"]]
input=list()
for i in range(0,8):
    i1,i2,i3 = input_train["y1"][i], input_train["y2"][i], bin_y3[i] 
    input.append([i1,i2,i3])

print(input,bin_out)
igy1 = mutual_info_classif(np.array(input),bin_out,discrete_features=True)
print("\nIG={}".format(igy1))

igy10 = mutual_info_classif(np.array([[0,2,0],[0,2,1]]),[0,1],discrete_features=True)
print("\nIGY10={}".format(igy10))
