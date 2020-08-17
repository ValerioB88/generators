import numpy as np
import torch
class SaltPepperNoiseFixation():
    def __init__(self, type_noise='pepper1', strength=0.2, size_canvas=(224, 224)):
        self.strength = strength # from 0 to 1
        self.size_canvas = size_canvas
        if type_noise == 'pepper1':
            self.value = 0
            self.fun = lambda x: torch.rand_like(x[0]) < self.distance_frame * self.strength
        elif type_noise == 'pepper2':
            # factor = 56 is the amount of pixel corresponding to 2 degrees with a canvas of 224 (we set 224 corresponding to 8 degrees)
            # We set the formula as E_2 / (E_2 + E) according to wikipedia, where E_2 is a constant factor approximately equal to 2 degrees
            # However consider that this drop in visual acuity is only marginally modelled by pepper noise. It makes sense to decrease/increase the rate of the hyperbola as needed. To do that we use another constant.
            # The function here is expressed in PIXEL.
            self.value = 0
            factor = 56
            self.fun = lambda x: torch.rand_like(x[0]) < 1 - (factor / (factor + (self.strength * self.distance_frame * 224 / 2))) # the two here is another constant multiplicative factor

        elif type_noise == 'salt':
            self.value = 1
        else:
            assert False, f"Salt Pepper Noise Type {type_noise} not recognized, chose [salt], [pepper], [pepper2]"

        xx, yy = np.meshgrid(range(self.size_canvas[0]), range(self.size_canvas[1]))

        distance_frame = np.array([np.linalg.norm([x - self.size_canvas[0] / 2, y - self.size_canvas[1] / 2])
                                   for x, y in zip(xx.flatten(), yy.flatten())]).reshape(self.size_canvas)
        distance_frame /= np.max(distance_frame)
        self.distance_frame = torch.tensor(distance_frame)



    def __call__(self, samples, factor=56):
        add_pepper = self.fun(samples)
        samples[:, add_pepper] = self.value
        return samples
