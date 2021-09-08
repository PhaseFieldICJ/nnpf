import numpy as np # FIXME: PyTorch instead ?!


__all__ = ["TensorBoardScalars"]


class TensorBoardScalars:
    """ Read scalars from TensorBoard event file

    Parameters
    ----------
    path: str
        Event file path or folder path containing one or more event files.

    Examples
    --------

    Training:
    >>> from nnpf.trainer import Trainer
    >>> from nnpf.models import Reaction
    >>> trainer = Trainer(default_root_dir="logs_doctest", name="Reaction", version="test_tbs", max_epochs=10, log_every_n_steps=1)
    >>> model = Reaction(train_N=10, val_N=20, seed=0, num_workers=4)
    >>> import contextlib, io
    >>> with contextlib.redirect_stdout(io.StringIO()):
    ...     with contextlib.redirect_stderr(io.StringIO()):
    ...         trainer.fit(model)

    Reading events:
    >>> import os
    >>> tbs = TensorBoardScalars(os.path.join("logs_doctest", "Reaction", "test_tbs"))
    >>> sorted(tbs.scalars)
    ['epoch', 'hp_metric', 'train_loss', 'val_loss']
    >>> wall_time, rel_time, step, value = tbs["val_loss"]

    >>> import numpy as np
    >>> np.set_printoptions(precision=6)
    >>> value
    array([1.655346, 1.63327 , 1.611218, 1.5892  , 1.567225, 1.545301,
           1.523438, 1.501644, 1.479929, 1.458302])
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

