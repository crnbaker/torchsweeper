import torch
import time
import numpy as np


class Timer:
    """
    Timing class with PyTorch CUDA compatibility.
    Create instance: ```timer.Time(n_iterations)```.
    Time an operation: func_output = ```timer.time(func)(*args)```.
    Get timing results: timings = ```timer.timings```.

    Args:
        n_iterations (int): Number of times to run the function
        cuda (bool): If ```True```, timings are determined on the GPU with ```torch.cuda.event``` calls. If ```False```,
            timings are determined using ```time.time()```. Set to ```True``` if planning to use the Timer to time a
            function operating on CUDA PyTorch Tensors.

    """
    def __init__(self, n_iterations, cuda=True):
        self.n_iterations = n_iterations
        self.cuda = cuda
        self.timings = []

    def time(self, func):
        """
        Return a timed version of a callable.
        Args:
            func: The callable to time the operation of.
        Returns:
            A new callable with the same arguments, returns and functionality of ```func```, that runs ```func```
                ```n_iterations``` times each time it is called, and logs the timings of each run to ```self.timings```.
        """
        def timed(*args):
            for n in range(self.n_iterations):
                if self.cuda:
                    t1 = torch.cuda.Event(enable_timing=True)
                    t2 = torch.cuda.Event(enable_timing=True)
                    t1.record()
                else:
                    t1 = time.time()

                result = func(*args)

                if self.cuda:
                    t2.record()
                    torch.cuda.synchronize()
                    elapsed = t1.elapsed_time(t2)
                else:
                    elapsed = (time.time() - t1) * 1e3
                self.timings.append(elapsed)
            return result
        return timed


class ParameterSweeper:
    """
    Time a callable multiple times, with different input arguments each time.
    Args:
        n_iterations (int): Number of times to run the function at each input argument configuration. Only the duration
            of the fastest call is stored.
        cuda (bool): If ```True```, timings are determined on the GPU with ```torch.cuda.event``` calls. If ```False```,
            timings are determined using ```time.time()```. Set to ```True``` if planning to use the Timer to time a
            function operating on CUDA PyTorch Tensors.

    """
    def __init__(self, n_iterations, cuda=True):
        self.timings = []
        self.cuda = cuda
        self.n_iterations = n_iterations

    def sweep(self, func):
        """
        Return a parameter swept version of a callable.
        Args:
            func: The callable to be parameter swept.
        Returns:
            A new callable with the same functionality as ```func```, but which accepts lists of each of its input
                arguments. ```func``` will be called with each of the input arguments lists in the lists. For example,
                to time the operation of ```np.add(1, 1)```, ```np.add(2, 2)``` and ```np.add(3, 3)``` you would first
                make a ParameterSweeper object: ```sweeper = ParameterSweeper(n_iters)``` and then use like this:
                ```sweeper.sweep(np.add)([1, 2, 3], [1, 2, 3])```. Any input arguments that are length 1 lists will be
                kept constant, so ```sweeper.sweep(np.add)([1, 2, 3], [1])``` is equivalent to
                ```sweeper.sweep(np.add)([1, 2, 3], [1, 1, 1])```.
        """
        def swept(*arglists):
            # Check same number of each argument
            swept_arglists = [arglist for arglist in arglists if len(arglist) > 1]
            constant_arg_count = len(set([len(swept_arglists) for swept_arglists in swept_arglists])) == 1
            if not constant_arg_count:
                raise ValueError('Lists of input argument values to sweep must be the same length or length 1.')
            n_test_value_sets = len(swept_arglists[0])
            arglists = [arglist * n_test_value_sets if len(arglist) == 1 else arglist for arglist in arglists]
            n_arguments = len(arglists)
            results = []

            for n in range(n_test_value_sets):
                timer = Timer(self.n_iterations, self.cuda)
                args = [arglists[i][n] for i in range(n_arguments)]
                results.append(timer.time(func)(*args))
                self.timings.append(np.min(timer.timings))
            return results
        return swept
