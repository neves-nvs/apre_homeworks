# Imports definition
import numpy as np

# Constants definition
LEARNING_RATE = 0.1


# ----------------------------------------------------- INIT FUNC ---------------------------------------------------- #


def activation_function(matrix):
    """
    * Applies activation function to each element in input matrix
    :param matrix: matrix to be updated
    :return: matrix with updated values
    """
    return np.tanh(matrix)


def derived_activation_function(matrix):
    """
    * Applies derived activation function to each element in input matrix
    :param matrix: matrix to be updated
    :return: matrix with updated values
    """
    return np.subtract(np.ones(matrix.shape), np.power(np.tanh(matrix), 2))


def derived_error_function(output_matrix, target_matrix):
    """
    * Applies derived squared error loss function to input elements
    :param output_matrix: obtained output
    :param target_matrix: expected output
    :return: matrix with error between output and target
    """
    return np.subtract(output_matrix, target_matrix)

# ----------------------------------------------------- INIT VARS ---------------------------------------------------- #


target = np.array([
    [1],
    [-1]
])

x_0 = np.array([
    [1],
    [1],
    [1],
    [1],
    [1]
])

w_1 = np.array([
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1]
])

b_1 = np.array([
    [1],
    [1],
    [1]
])

w_2 = np.array([
    [1, 1, 1],
    [1, 1, 1]
])

b_2 = np.array([
    [1],
    [1]
])

w_3 = np.array([
    [0, 0],
    [0, 0]
])

b_3 = np.array([
    [0],
    [0]
])

# Is put in an array to be used in loops
w = [0, w_1, w_2, w_3]  # The initial zero is just to align layer nº with the index of the array (is basically clutter)
b = [0, b_1, b_2, b_3]  # The initial zero is just to align layer nº with the index of the array (is basically clutter)

# Hold outputs without activation function
z = [0]  # The initial zero is just to align layer nº with the index of the array (is basically clutter)

# Hold output with activation function
x = [x_0]

# Holds delta errors for each layer
d = [0]  # The initial zero is just to align layer nº with the index of the array (is basically clutter)


# ------------------------------------------------- OUTPUT PROCESSING ------------------------------------------------ #


print("------------------------------- OUTPUT PROCESSING -------------------------------")

# Applies forward propagation to each layer of the matrix and calculates output and activated output
for i in range(1, 4):

    print(f"################## Layer nº: {i} ##################")
    print(f"W[{i}]: \n{w[i]}")
    print(f"X[{i - 1}]: \n{x[i - 1]}")
    print(f"B[{i}]: \n{b[i]}")

    # Calculates both outputs
    out = np.add(np.matmul(w[i], x[i - 1]), b[i])
    out_act = activation_function(out)

    print(f"Out -> Z[{i}]: \n{out}")
    print(f"Out Act -> X[{i}]: \n{out_act}")

    # Stores them for later use
    z.append(out)
    x.append(out_act)


# ------------------------------------------------ DELTA CALCULATION ------------------------------------------------- #


print("------------------------------- DELTA CALCULATION -------------------------------")

# Calculates all deltas
for i in range(3, 0, -1):

    print(f"################## Delta nº: {i} ##################")

    # When delta index is equal to the number of the last layer, we use a different formula
    if i == 3:

        print(f"X[{i}]: \n{x[i]}")
        print(f"Z[{i}]: \n{z[i]}")

        # Calculates both multiplication parameters and prints them
        error = derived_error_function(x[i], target)
        act = derived_activation_function(z[i])
        print(f"Error matrix: \n{error}")
        print(f"Act matrix: \n{act}")

        # Calculates delta for the last layer, prints and appends it to be used in further calculations
        result = np.multiply(error, act)
        print(f"Resulting delta: \n{result}")
        d.append(result)

    # Since it is not the last layer, we need to use the previous layer's delta to calculate the current's
    else:

        print(f"W[{i + 1}]: \n{w[i + 1]}")
        print(f"Z[{i}]: \n{z[i]}")
        print(f"D[{i + 1}]: \n{d[1]}")

        # Performs both multiplication parameters and prints them
        error = np.matmul(w[i + 1].transpose(), d[1])
        act = derived_activation_function(z[i])
        print(f"Error matrix: \n{error}")
        print(f"Act matrix: \n{act}")

        # Calculates delta for the last layer, prints and appends it to be used in further calculations
        result = np.multiply(error, act)
        print(f"Resulting delta: \n{result}")
        d.insert(1, result)


# ------------------------------------------------- UPDATE WEIGHTS --------------------------------------------------- #


print("------------------------------- UPDATE WEIGHTS -------------------------------")

# Goes through every layer and update both weights and biases with the previous deltas
for i in range(1, 4):

    print(f"################## Layer nº: {i} ##################")
    print(f"W_old[{i}]: \n{w[i]}")
    print(f"B[{i}]: \n{b[i]}")
    print(f"D[{i}]: \n{d[i]}")
    print(f"X[{i - 1}]: \n{x[i - 1]}")

    # Performs the update formula for both w and b
    w_offset = LEARNING_RATE * (np.multiply(d[i], x[i - 1].transpose()))
    b_offset = LEARNING_RATE * d[i]
    w[i] = np.subtract(w[i], w_offset)
    b[i] = np.subtract(b[i], b_offset)

    print(f"W Offset: \n{w_offset}")
    print(f"B Offset: \n{b_offset}")
    print(f"New W: \n{w[i]}")
    print(f"New B: \n{b[i]}")
