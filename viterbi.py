import numpy as np



def viterbi(observed_seq, transition_probs, initial_probs, emission_probs):
    T = [dict([(s, 0) for s in transition_probs]) for _ in range(len(observed_seq))]
    T[0] = dict([(i, np.log(initial_probs[i]) + emission_probs[i][observed_seq[0]]) 
                 for i in emission_probs])
    T_bp = [dict([(s, 0) for s in transition_probs]) for _ in range(len(observed_seq))]
    T_bp[0] = dict([(i, i) for i in emission_probs])
    
    for k in range(1, len(observed_seq)):
        for j in transition_probs:
            max_val = max(transition_probs[j], key=lambda k: transition_probs[j][k])
            for i in transition_probs:
                if j in transition_probs[i]:
                    max_val = max(
                        max_val, 
                        T[k - 1][i] + transition_probs[i][j] + emission_probs[j][observed_seq[k]])
            T_bp[k][j] = max_val

            prev_state = T_bp[k][j]
            T[k][j] = (
                T[k - 1][prev_state] + 
                transition_probs[prev_state][j] + 
                emission_probs[j][observed_seq[k]]
            )

    state_sequence = [None for _ in range(len(observed_seq))]
    state_sequence[-1] = max(transition_probs, key=lambda k: T[-1][k])

    for i in range(len(observed_seq) - 1):
        i = len(observed_seq) - 2 - i 
        state_sequence[i] = T_bp[i + 1][state_sequence[i + 1]]

    return state_sequence