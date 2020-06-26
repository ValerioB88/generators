from multiprocessing.dummy import freeze_support
from torch.utils.data import DataLoader
import numpy as np
from generate_datasets.generators.utils_generator import TranslationType, BackGroundColorType
from generate_datasets.generators.dummy_face_generators.dummy_faces_random_generator import DummyFaceRandomGenerator
from generate_datasets.generators.translate_generator import TranslateGenerator
from generate_datasets.generators.leek_generator import LeekGenerator
import visualization.vis_utils as vis

class FixedTranslationGenerator(TranslateGenerator):
    def __init__(self, grid_size, num_repetitions):
        self.grid_size = grid_size
        self.num_repetitions = num_repetitions
        self.mesh_trX, self.mesh_trY, self.mesh_class, self.mesh_rep = [], [], [], []
        self.current_index = 0

    def _finalize_init_(self):
        for groupID in range(self.num_classes):
            minX, maxX, minY, maxY = self.translations_range[groupID]
            x, y, c, r = np.meshgrid(np.arange(minX, maxX, self.grid_size),
                                     np.arange(minY, maxY, self.grid_size),
                                     groupID,
                                     np.arange(self.num_repetitions))
            self.mesh_trX.extend(x.flatten())
            self.mesh_trY.extend(y.flatten())
            self.mesh_class.extend(c.flatten())
            self.mesh_rep.extend(r.flatten())

        self.save_stats()
        self.current_index = 0

    def __len__(self):
        return len(self.mesh_trX)

    def _get_translation_(self, label=None, image_name=None):
        self.current_index += 1
        return self.mesh_trX[self.current_index], self.mesh_trY[self.current_index]


    def _get_label_(self):
        return self.mesh_class[self.current_index]


