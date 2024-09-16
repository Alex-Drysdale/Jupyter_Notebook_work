import matplotlib.pyplot as plt
from IPython import display

plt.ion() # enables interactive mode in matplotlib

def plot(scores, mean_scores):
    display.clear_output(wait=True) # clear previous plot output
    display.display(plt.gcf()) # get current figure to display

    plt.clf() # clear the current figure

    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')

    plt.ylim(ymin=0)

    plt.text(len(scores)-1, scores[-1], str(scores[-1])) # display the last score and mean score as text on the graph
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    plt.legend()
    plt.pause(0.1) # small pause for refresh; tune this value for smoother updates