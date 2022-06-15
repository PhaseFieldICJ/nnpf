"""
Tools for slicing and reconstruction of 2D shapes
"""

import torch

__all__ = [
    "slice_shape",
    "slices_to_mask",
]

def slice_shape(shape, bounds, positions, axis=1, step=1e-4, threshold=0., inside=-1.):
    """ Slice 2D shape at given positions for given axis

    Parameters
    ----------
    shape: function
        2D shape to be sliced
    bounds: iterable of pairs of float
        Bounds where the shape is defined (bounds must be outside the shape)
    positions: iterable of float
        Slice positions
    axis: int
        Axis (dimension) where the slice positions are defined
    step: float
        Space step used when searching for interface crossing
    threshold: float
        Value at the interface (eg 0 for signed distance, 0.5 for phase fields)
    inside: float
        reference value for an inside point (only the sign of `inside - threshold` matters)

    Returns
    -------
    slices: list of tuple of list
        Each list item map to a input position and contains the positions
        of interface crossing and the bounds.
    """

    X_along = torch.arange(bounds[1 - axis][0], bounds[1 - axis][1], step)
    X = [None] * 2
    X[1 - axis] = X_along

    slices = []

    for pos in positions:
        X[axis] = torch.full_like(X_along, pos)
        values = (shape(*X) - threshold) * (inside - threshold)
        assert values[0] < 0 and values[-1] < 0, "Bounds must be outside the shape!"
        crossing = values[:-1] * values[1:]

        curr_slice = [X_along[0]]
        state = "outside"
        for c in torch.nonzero(crossing <= 0).squeeze():
            # Crossing position
            if crossing[c] < 0:
                crossing_pos = X_along[c] + step / 2
            elif values[c] == 0:
                crossing_pos = X_along[c]
            else:
                crossing_pos = X_along[c + 1]

            # State update
            if state == "outside":
                curr_slice.append(crossing_pos)
                state = "inside"
            elif values[c + 1] > 0:
                curr_slice.append(crossing_pos)
                state = "outside"

        curr_slice.append(X_along[-1])
        if axis == 0:
            slices.append((torch.as_tensor(curr_slice), pos))
        else:
            slices.append((pos, torch.as_tensor(curr_slice)))

    return slices

def slices_to_mask(slices, domain, shrink=0):
    """ Creates inside/outside mask from slices

    Parameters
    ----------
    slices: list of list
        Each list item map to a input position and contains the positions
        of interface crossing and the bounds (see `slice_shape`)
    domain: nnpf.domain.Domain
        Discretization domain
    shrink: float
        Shrink each interval by given length on each side (interval ignored if empty)

    Returns
    -------
    inside, outside: torch.Tensor
        inside and outside masks for given slices
    """
    inside = torch.full(domain.N, False, dtype=torch.bool, device=domain.device)
    outside = torch.full(domain.N, False, dtype=torch.bool, device=domain.device)

    for s in slices:
        s = tuple(torch.as_tensor(x) for x in s)
        if s[0].numel() == 1:
            axis = 1
        elif s[1].numel() == 1:
            axis = 0
        else:
            raise ValueError(f"Invalid slice {s}!")

        for i in range(len(s[axis]) - 1):
            a = s[axis][i] + shrink
            b = s[axis][i + 1] - shrink
            if a > b:
                continue

            idx = domain.index(s[1 - axis], [a, b])
            interval = (idx[0], slice(idx[1][0], idx[1][1] + 1))
            if axis == 0:
                interval = interval[::-1]

            if i % 2 == 0:
                outside[interval] = True
            else:
                inside[interval] = True


    # Avoid overlapping
    outside[inside] = False

    return inside, outside

