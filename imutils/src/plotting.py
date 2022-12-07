# Function to help plotting

def plot_stimuli(ax, stimulus_start, stimulus_length, **kwargs):
    """This function was originally (and is still there) in epifluorescence_calcium_imaging/dev/test_measure_results_from_fiji.py"""
    for start in stimulus_start:
        ax.axvspan(start, start+stimulus_length, 0, 1, **kwargs)