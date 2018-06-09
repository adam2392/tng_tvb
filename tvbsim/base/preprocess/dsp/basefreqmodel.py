from fragility.base.constants.config import Config
from fragility.base.utils.log_error import initialize_logger
from fragility.base.model.basemodel import BaseWindowModel
import numpy as np


class BaseFreqModel(BaseWindowModel):
    def __init__(self, winsizems, stepsizems, samplerate):
        BaseWindowModel.__init__(self, winsizems=winsizems,
                                 stepsizems=stepsizems,
                                 samplerate=samplerate)

    def buffer(self, x, n, p=0, opt=None):
        '''Mimic MATLAB routine to generate buffer array

        MATLAB docs here: https://se.mathworks.com/help/signal/ref/buffer.html

        Args
        ----
        x:   signal array
        n:   number of data segments
        p:   number of values to overlap
        opt: initial condition options. default sets the first `p` values
             to zero, while 'nodelay' begins filling the buffer immediately.
        '''
        if p >= n:
            raise ValueError('p ({}) must be less than n ({}).'.format(p, n))
        assert isinstance(n, int)
        assert isinstance(p, int)

        # Calculate number of columns of buffer array
        if p == 0:
            cols = int(np.floor(len(x) / float(n - p)))
        else:
            cols = int(np.floor(len(x) / float(p)))

        # Check for opt parameters
        if opt == 'nodelay':
            # Need extra column to handle additional values left
            cols -= 1
        elif opt is not None:
            raise SystemError('Only `None` (default initial condition) and '
                              '`nodelay` (skip initial condition) have been '
                              'implemented')
        # Create empty buffer array. N = size of window X # cols
        b = np.zeros((n, cols))

        # print("bshape is: ", b.shape)
        # Fill buffer by column handling for initial condition and overlap
        j = 0
        for i in range(cols):
            # Set first column to n values from x, move to next iteration
            if i == 0 and opt == 'nodelay':
                b[0:n, i] = x[0:n]
                continue
            # set first values of row to last p values
            elif i != 0 and p != 0:
                b[:p, i] = b[-p:, i - 1]
            # If initial condition, set p elements in buffer array to zero
            else:
                b[:p, i] = 0
            # Assign values to buffer array from x
            b[p:, i] = x[p * (i + 1):p * (i + 2)]

        return b
