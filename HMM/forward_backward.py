import numpy as np

from .utils import  log_sum_exp



def forward(observed_seq, transition_probs, emission_probs, initial_probs):
    alpha = [dict([(s, 0) for s in transition_probs]) for _ in range(len(observed_seq))]
    alpha[0] = dict([
        (i, np.log(initial_probs[i]) + emission_probs[i][observed_seq[0]]) 
            for i in emission_probs])
    for k in range(1, len(observed_seq)):
        for i in transition_probs:
            alpha[i][k] = emission_probs[i][observed_seq[k]] + log_sum_exp([
                alpha[j][k-1] + transition_probs[i][j] for j in transition_probs[i]])
    return alpha        




def backward(observed_seq, transition_probs, emission_probs):
    beta = [dict([(s, 0.0) for s in transition_probs]) for _ in range(len(observed_seq))]

    for j in transition_probs:
        beta[j][len(observed_seq) - 1] = 0

    for k in range(len(observed_seq) - 2, -1, -1):
        for i in transition_probs:
            beta[i][k] = log_sum_exp([
                beta[j][k + 1] + transition_probs[i][j] + emission_probs[j][observed_seq[k + 1]]
                for j in transition_probs[i]
            ])
    return beta
