"""
Splitting methods

"""

class Lie:
    """ Lie splitting method A(t)B(t) """

    def __init__(self, A, B, t):
        self.A = A(t)
        self.B = B(t)

    def __eval__(self, x):
        return self.A(self.B(x))


class Strang:
    """ Strang splitting method A(t/2)B(t)A(t/2) """

    def __init__(self, A, B, t):
        self.A = A(t/2)
        self.B = B(t)

    def __eval__(self, x):
        return self.A(self.B(self.A(x)))


