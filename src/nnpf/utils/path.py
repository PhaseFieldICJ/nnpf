__all__ = [
    "fix_path",
    "checkpoint_from_path",
]


def fix_path(path):
    """ Convert path from Posix to/from Windows filesystems """
    # Naive way to find out if given path if from Windows
    # No really reliable since \\ may come from a valid POSIX path...
    is_windows = path.find('\\') != -1

    # Returns given path if on right filesystem
    import os
    if (is_windows and os.sep == '\\') or (not is_windows and os.sep == '/'):
        return path

    import pathlib
    if is_windows:
        return pathlib.PureWindowsPath(path).as_posix() # Windows -> POSIX
    else:
        return str(pathlib.PureWindowsPath(path)) # POSIX -> Windows


def checkpoint_from_path(checkpoint_path):
    """
    Returns path if it points to an actual file,
    otherwise search for the last checkpoint of the form
    "path/checkpoints/epoch=*.ckpt"
    """
    import os
    checkpoint_path = fix_path(checkpoint_path)

    # If path if a folder, found last checkpoint from checkpoints subfolder
    if os.path.isdir(checkpoint_path):
        dirname = os.path.join(os.path.expanduser(checkpoint_path), r"checkpoints")

        # Don't get the last.ckpt automatically since it is not the best checkpoint
        #if os.path.exists(os.path.join(dirname, r"last.ckpt")):
        #    return os.path.join(dirname, r"last.ckpt")

        import glob
        import re
        glob_expr = os.path.join(dirname, r"epoch=*.ckpt")
        checkpoint_list = glob.glob(glob_expr)
        if len(checkpoint_list) > 0:
            checkpoint_path = sorted(checkpoint_list, key=lambda s: tuple(int(v) for v in re.findall(r"(\d+)", s)))[-1]

    return checkpoint_path

