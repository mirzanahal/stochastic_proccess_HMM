import numpy as np


def gamma(observed_seq, transition_probs, forward_probs, backward_probs):
    gamma_probs = [dict([(s, 0) for s in transition_probs]) for _ in range(len(observed_seq))]

    for k in range(len(observed_seq)):
        denomenator = sum(
            [np.exp( 1000 + forward_probs[i][k] + backward_probs[i][k]) for i in transition_probs])
        for i in transition_probs:
            gamma_probs[k][i] = np.exp( 1000 + forward_probs[i][k] + backward_probs[i][k]) / denomenator

    return gamma_probs



def zeta(observed_seq, transition_probs, emission_probs, forward_probs, backward_probs):
    zeta_probs = [dict([(s, dict([(s, 0) for s in transition_probs])) for s in transition_probs]) 
                for _ in range(len(observed_seq))]
    
    for k in range(len(observed_seq) - 1):
        denomenator = sum([sum([
            np.exp( 1000 + 
                forward_probs[i][k] + transition_probs[i][j] + 
                backward_probs[j][k+1] + emission_probs[j][observed_seq[k + 1]])
            for j in transition_probs[i]]) for i in transition_probs])
            
        for i in transition_probs:
            for j in transition_probs[i]:
                if j in transition_probs[i]:
                    zeta_probs[k][i][j] = np.exp( 1000 + 
                            forward_probs[i][k] + transition_probs[i][j]+ 
                            backward_probs[j][k + 1] + emission_probs[j][observed_seq[k + 1]]
                    ) / denomenator
    return zeta_probs


'a matrix'
def estimate_transition_probs(observed_seq, transition_probs, gamma_probs, zeta_probs):
    a = dict([(s1, dict([(s2, 0) for s2 in transition_probs[s1]])) for s1 in transition_probs])
          
    for i in transition_probs:
        for j in transition_probs[i]:
            a[i][j] = 0
            denomenator_a = 0
            for k in range(len(observed_seq)-1):
                a[i][j] += zeta_probs[k][i][j]
                denomenator_a += gamma_probs[k][i]

            if (denomenator_a == 0):
                a[i][j] = 0
            else:
                a[i][j] = a[i][j]/denomenator_a
    
    return a


'b matrix'
def estimate_emission_probs(
        observed_seq, transition_probs, emission_probs, gamma_probs, observation_names):
    
    b = dict([(s, dict([(o, 0) for o in emission_probs[s]])) for s in emission_probs])
    indices = {}
    for o in observation_names:
        indices[o] = [idx for idx, val in enumerate(observed_seq) if val == o]

    for i in transition_probs: 
        for o in range(len(observation_names)): 
            numerator_b = sum([gamma_probs[k][i] for k in indices[observed_seq[o]]])
            denomenator_b = sum([gamma_probs[k][i] for k in observed_seq])

            if (denomenator_b == 0):
                b[i][o] = 0
            else:
                b[i][o] = numerator_b / denomenator_b
    return b


def baum_welch(
        observed_seq, transition_probs, emission_probs, observation_names, 
        forward_probs, backward_probs):

    gamma_probs = gamma(
        observed_seq=observed_seq,
        transition_probs=transition_probs,
        forward_probs=forward_probs,
        backward_probs=backward_probs
    )

    zeta_probs = zeta(
        observed_seq=observed_seq, 
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        forward_probs=forward_probs,
        backward_probs=backward_probs
    )

    a = estimate_transition_probs(
        observed_seq, transition_probs, gamma_probs, zeta_probs)
    b = estimate_emission_probs(
        observed_seq, transition_probs, emission_probs, gamma_probs, observation_names)        

    transition_probs = a
    emission_probs = b

    return transition_probs, emission_probs    