import torch
import time
import numpy as np


class Timer:
    def __init__(self, n_iterations, cuda=True):
        self.n_iterations = n_iterations
        self.cuda = cuda
        self.timings = []

    def time(self, func):
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
    def __init__(self, n_iterations, cuda=True):
        self.timings = []
        self.cuda = cuda
        self.n_iterations = n_iterations

    def sweep(self, func):
        def swept(*arglists):
            # Check same number of each argument
            if len(set([len(arglist) for arglist in arglists])) > 1:
                raise ValueError('Same number of each argument must be provided to ParameterScanner')
            N_test_value_sets = len(arglists[0])
            N_arguments = len(arglists)
            results = []

            for n in range(N_test_value_sets):
                timer = Timer(self.n_iterations, self.cuda)
                args = [arglists[i][n] for i in range(N_arguments)]
                results.append(timer.time(func)(*args))
                self.timings.append(np.min(timer.timings))
            return results
        return swept
