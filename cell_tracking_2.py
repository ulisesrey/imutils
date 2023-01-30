# First part of the section 1. of the napari tracking tutorial
# https://napari.org/stable/tutorials/tracking/index.html

import btrack

objects = btrack.dataio.import_CSV('/Users/ulises.rey/local_data/napari_example.csv')

with btrack.BayesianTracker() as tracker:

    # configure the tracker using a config file
    tracker.configure_from_file('cell_config.json')

    tracker.append(objects)
    tracker.volume=((0,1600), (0,1200), (-1e5,1e5))

    # track and optimize
    tracker.track_interactive(step_size=100)
    tracker.optimize()

    # get the tracks in a format for napari visualization
    data, properties, graph = tracker.to_napari(ndim=2)