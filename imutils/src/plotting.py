# Function to help plotting
def consecutive_count(arr):
    count = 1
    start_indexes = []
    counts = []
    start_index = 0

    for i in range(1, len(arr)):
        if arr[i] == arr[i-1]:
            count += 1
        else:
            start_indexes.append(start_index)
            counts.append(count)
            start_index = i
            count = 1

    start_indexes.append(start_index)  # to include the last sequence of consecutive numbers
    counts.append(count)

    return start_indexes, counts

def plot_stimuli(ax, stimulus_start, stimulus_length, **kwargs):
    """This function was originally (and is still there) in epifluorescence_calcium_imaging/dev/test_measure_results_from_fiji.py"""
    #TODO: THis version is too simple, would be good to improve to have different lengths.
    for i, start in enumerate(stimulus_start):
        ax.axvspan(start, start+stimulus_length[i], 0, 1, **kwargs)

