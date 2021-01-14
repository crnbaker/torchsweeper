from torchsweeper import ParameterSweeper
from torchsweeper.signals import NCyclePulse, SignalImage, LengthSweepImageSet


# Function to sweep MONAI and Scipy methods with different Tensor sizes
def sweep_signal_lengths(callable, constants=[], im_type='numpy', min_len_pow_2=2, max_len_pow_2=5, n_iters=5, sigs_per_im=1):

    snr = 10
    pulse = NCyclePulse(1e6, 3, snr)
    image = SignalImage(signal=pulse, stacking_shape=(sigs_per_im,), use=im_type)
    image_set = LengthSweepImageSet(image=image, min_len=min_len_pow_2, max_len=max_len_pow_2)

    # Run parameter sweep on callable
    sweeper = ParameterSweeper(n_iters, cuda=False)
    output = sweeper.sweep(callable)(
        image_set(),
        *[[arg] for arg in constants]
    )

    return output, sweeper, image_set.signal_lengths