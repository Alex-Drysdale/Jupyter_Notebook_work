import numpy as np
import matplotlib.pyplot as plt

# Initial states
initial_states = [0, 3, 0, 0, 0, 0]

# Parameters (example values, adjust as necessary)
transcription_rate = 50
basal_transcript_rate = 0.01
mRNA_degradation_rate = 1
translation_rate = 5
protein_degradation_rate = 0.1
n = 2  # Hill coefficient

# Gillespie function
def gillespie(initial_states, transcription_rate, basal_transcript_rate, mRNA_degradation_rate, translation_rate, protein_degradation_rate, n, max_time):
    t = 0
    state = np.array(initial_states)
    times = [t]
    states = [state.copy()]

    while t < max_time:
        TetR_mRNA, TetR_protein, cI_mRNA, cI_protein, LacI_mRNA, LacI_protein = state

        # Propensities
        a = np.array([
            (transcription_rate / (1 + LacI_protein ** n)) + basal_transcript_rate,  # production of TetR_mRNA
            mRNA_degradation_rate * TetR_mRNA,  # degradation of TetR_mRNA
            translation_rate * TetR_mRNA,  # production of TetR protein
            protein_degradation_rate * TetR_protein,  # degradation of TetR protein

            (transcription_rate / (1 + TetR_protein ** n)) + basal_transcript_rate,  # production of cI_mRNA
            mRNA_degradation_rate * cI_mRNA,  # degradation of cI_mRNA
            translation_rate * cI_mRNA,  # production of cI protein
            protein_degradation_rate * cI_protein,  # degradation of cI protein

            (transcription_rate / (1 + cI_protein ** n)) + basal_transcript_rate,  # production of LacI_mRNA
            mRNA_degradation_rate * LacI_mRNA,  # degradation of LacI_mRNA
            translation_rate * LacI_mRNA,  # production of LacI protein
            protein_degradation_rate * LacI_protein  # degradation of LacI protein
        ])

        a0 = np.sum(a)
        if a0 == 0:
            break

        r1 = np.random.random()
        tau = (1 / a0) * np.log(1 / r1)

        r2 = np.random.uniform(0, 1)
        cumulative_sum = np.cumsum(a)
        reaction_index = np.searchsorted(cumulative_sum, r2 * a0)

        # Update the system
        state_changes = [
            [1, 0, 0, 0, 0, 0],   # mRNA_TetR production
            [-1, 0, 0, 0, 0, 0],  # mRNA_TetR degradation
            [0, 1, 0, 0, 0, 0],   # protein_TetR production
            [0, -1, 0, 0, 0, 0],  # protein_TetR degradation

            [0, 0, 1, 0, 0, 0],   # mRNA_cI production
            [0, 0, -1, 0, 0, 0],  # mRNA_cI degradation
            [0, 0, 0, 1, 0, 0],   # protein_cI production
            [0, 0, 0, -1, 0, 0],  # protein_cI degradation

            [0, 0, 0, 0, 1, 0],   # mRNA_LacI production
            [0, 0, 0, 0, -1, 0],  # mRNA_LacI degradation
            [0, 0, 0, 0, 0, 1],   # protein_LacI production
            [0, 0, 0, 0, 0, -1],  # protein_LacI degradation
        ]

        state += state_changes[reaction_index]
        t += tau

        times.append(t)
        states.append(state.copy())

    return np.array(times), np.array(states)


# Simulation parameters
max_time = 1000
