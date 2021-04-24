import numpy as np # FIXME: PyTorch instead ?!


__all__ = ["TensorBoardScalars"]


class TensorBoardScalars:
    """ Read scalars from TensorBoard event file

    Parameters
    ----------
    path: str
        Event file path or folder path containing one or more event files.
    """

    def __init__(self, path):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        self.events = EventAccumulator(path)
        self.reload()

    def reload(self):
        """ Loads new events """
        self.events.Reload()

    @property
    def scalars(self):
        """ Available scalars """
        return set(self.events.Tags()['scalars'])

    def __getitem__(self, name):
        """ Get scalar evolution

        Parameters
        ----------
        name: str
            Scalar name

        Returns
        -------
        wall_time: numpy.ndarray
            Wall time of each sample
        rel_time: numpy.ndarray
            Relative time (to first event's timestamp) of each sample
        step: numpy.ndarray
            Step of each sample
        value: numpy.ndarray
            Value of each sample
        """
        wall_time, step, value = map(lambda d: np.asarray(d), zip(*self.events.Scalars(name)))
        rel_time = wall_time - self.events.FirstEventTimestamp()
        return wall_time, rel_time, step, value

