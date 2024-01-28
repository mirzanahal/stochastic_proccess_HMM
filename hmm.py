import numpy as np
from tqdm import tqdm

from .utils import log_probs, matrix_distance, log_sum_exp
from .forward_backward import forward, backward
from .baum_welch import baum_welch
from .viterbi import viterbi

class HiddenMarkovModel:
    def __init__(self, initial_probs, adjacency_mat, observation_names, emission_probs):
        self.initial_probs = initial_probs
        self.transition_probs = log_probs(adjacency_mat)
        self.observation_names = observation_names
        self.emission_probs = log_probs(emission_probs)

    def update_parameters(self, observed_seqs, iterations=100, verbose=True):
        for observed_seq in tqdm(observed_seqs):    
            for _ in range(iterations): 
                forward_probs = forward(
                    observed_seq=observed_seq,
                    transition_probs=self.transition_probs,
                    emission_probs=self.emission_probs,
                    initial_probs=self.initial_probs
                    )
                backward_probs = backward(
                    observed_seq=observed_seq,
                    transition_probs=self.transition_probs,
                    emission_probs=self.emission_probs
                )
                new_transition_probs, new_emission_probs = baum_welch(
                    observed_seq, self.transition_probs, self.emission_probs, self.observation_names, 
                    forward_probs, backward_probs
                )
                new_emission_probs = log_probs(new_emission_probs)
                new_transition_probs = log_probs(new_transition_probs)
                
                if verbose:
                    transition_update_distance = matrix_distance(
                        new_transition_probs, self.transition_probs)
                    print("Transition update distance:{}".format(
                        transition_update_distance
                    ))
                    
                    
                    emission_update_distance = matrix_distance(
                        new_emission_probs, self.emission_probs)
                    print("Emission update distance:{}".format(
                        emission_update_distance
                    ))
                    # print(new_emission_probs)
                    # print(self.emission_probs)

                self.transition_probs = new_transition_probs
                self.emission_probs = new_emission_probs

    def evaluate(self, observed_seqs, method='forward'):
        seq_probs = []
        for observed_seq in observed_seqs:
            if method == 'forward':
                probs = forward(
                    observed_seq=observed_seq,
                    transition_probs=self.transition_probs,
                    emission_probs=self.emission_probs,
                    initial_probs=self.initial_probs
                    )
                seq_prob = log_sum_exp(list(probs[-1].values()))
            else:
                probs = backward(
                    observed_seq=observed_seq,
                    transition_probs=self.transition_probs,
                    emission_probs=self.emission_probs
                )
                seq_prob = log_sum_exp([(
                    probs[0][k] + self.emission_probs[k][observed_seq[0]] + np.log(
                    self.initial_probs[k])) for k in self.transition_probs])
            seq_probs.append(seq_prob)
        return seq_probs
    

    def decode(self, observed_seq):
        path = viterbi(
            observed_seq=observed_seq,
            transition_probs=self.transition_probs,
            initial_probs=self.initial_probs,
            emission_probs=self.emission_probs
        )
        return path
                

