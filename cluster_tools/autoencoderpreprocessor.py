#Remove this later
import sys
sys.path.insert(0, '/Users/mishamesarcik/Workspace/phd/Workspace/lofar-dev')
import preprocessor


def preprocess(observation):
    if observation is None:
        raise ValueError('No data to preprocess.')

    if observation is None or 'visibilities' not in observation:
        raise ValueError('No visibilities in observation.')

    result = observation['visibilities'].transpose([0, 2, 1, 3])

    return result

