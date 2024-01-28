import json
import numpy as np

from pathlib import Path



def read_json(dir):
    data = json.loads(Path(dir).read_text())
    return data


def write_json(dict, dir):
    output = json.dumps(dict)
    f = open(dir, "w")
    f.write(output)
    f.close()


def log_sum_exp(arr):
    a = np.max(arr)
    return a + np.log(np.sum([np.exp(x - a) for x in arr]))


def log_probs(prob_matrix, eps=1e-17):
    log_prob_matrix = {}
    for i in prob_matrix:
        log_prob_matrix[i] = {}
        for j in prob_matrix[i]:
            log_prob_matrix[i][j] = np.log(prob_matrix[i][j]+ eps)
    return log_prob_matrix

def calculate_probability(log_prob_matrix):
    prob_matrix = {}
    for i in log_prob_matrix:
        prob_matrix[i] = {}
        for j in log_prob_matrix[i]:
            prob_matrix[i][j] = np.exp(log_prob_matrix[i][j])
    return prob_matrix


def dict_to_list(dictionary):
    _list = []
    for i in dictionary.values():
        _list += i.values()
    return np.array(_list)

def matrix_distance(matrix1, matrix2):
    matrix1 = dict_to_list(matrix1)
    matrix2 = dict_to_list(matrix2)

    if matrix1.shape != matrix2.shape:
        raise ValueError("Matrices must have the same shape {} vs. {}".format(
            matrix1.shape, matrix2.shape
        ))

    distance = np.linalg.norm(matrix1 - matrix2)
    return distance


def cast_keys_to_int(dictionary):
    return {int(key): value for key, value in dictionary.items()}
