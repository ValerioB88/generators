import numpy as np
import torch
class SaltPepperNoiseFixation():
    def __init__(self, type_noise='pepper', strength=0.2, size_canvas=(224, 224)):
        self.strength = strength # from 0 to 1
        self.size_canvas = size_canvas
        if type_noise == 'pepper':
            self.value = 0
        elif type_noise == 'salt':
            self.value = 1
        else:
            assert False, "Salt Pepper Noise Type not recognized, chose [salt] or [pepper]"

        xx, yy = np.meshgrid(range(self.size_canvas[0]), range(self.size_canvas[1]))

        distance_frame = np.array([np.linalg.norm([x - self.size_canvas[0] / 2, y - self.size_canvas[1] / 2])
                                   for x, y in zip(xx.flatten(), yy.flatten())]).reshape(self.size_canvas)
        distance_frame /= np.max(distance_frame)
        self.distance_frame = torch.tensor(distance_frame)

    def __call__(self, samples):
        add_pepper = torch.rand_like(samples[0]) < self.distance_frame * self.strength
        samples[:, add_pepper] = self.value
        return samples
