from multiprocessing.dummy import freeze_support
from torch.utils.data import DataLoader
import numpy as np
from generate_datasets.generators.utils_generator import TranslationType, BackGroundColorType
from generate_datasets.generators.dummy_face_generators.dummy_faces_random_generator import DummyFaceRandomGenerator
from generate_datasets.generators.translate_generator import TranslateGenerator
from generate_datasets.generators.leek_generator import LeekGenerator
import visualization.vis_utils as vis


class FiniteTranslationGeneratorMixin():
    """
        This mixin will disable the random generation, and it will generate samples at fixed locations of a canvas.
        If shuffle is disabled, and num_repetitions is 1, it will pass through each translation, for each class at the time.
        If num_repetitions = n, it will resample that class n times for each translation (useful when each object within a class is
        itself randomly sampled).
        This means that if a classes contains multiple and different elements (as it should), num_repetition should be set AT LEAST to that number of different elements. You won't have the guarantee that every element of that class is gonna be present for every translation. (I don't think it matters)
    """
    def __init__(self, grid_size, num_repetitions=1):
        self.grid_size = grid_size
        self.num_repetitions = num_repetitions
        self.mesh_trX, self.mesh_trY, self.mesh_class, self.mesh_rep = [], [], [], []
        self.current_index = 0

    def _finalize_init_(self):
        for groupID in range(self.num_classes):
            minX, maxX, minY, maxY = self.translations_range[groupID]
            # Recall that maxX should NOT be included (it's [minX, maxX)
            x, y, c, r = np.meshgrid(np.arange(minX, maxX, self.grid_size),
                                     np.arange(minY, maxY, self.grid_size),
                                     groupID,
                                     np.arange(self.num_repetitions))
            self.mesh_trX.extend(x.flatten())
            self.mesh_trY.extend(y.flatten())
            self.mesh_class.extend(c.flatten())
            self.mesh_rep.extend(r.flatten())
        print('Created Finite Dataset [{}] with {} elements'.format(self.name_generator, self.__len__()))

        self.save_stats()
        # self.current_index = 0

    def __len__(self):
        return len(self.mesh_trX)

    def _get_translation_(self, label=None, image_name=None, idx=None):
        return self.mesh_trX[idx], self.mesh_trY[idx]

    def _finalize_get_item_(self, canvas, label, more):
        canvas, label, more = super()._finalize_get_item_(canvas, label, more)
        # self.current_index += 1
        return canvas, label, more

    def _get_label_(self, item):
        return self.mesh_class[item]


