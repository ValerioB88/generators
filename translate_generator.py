from generate_datasets.generators.utils_generator import get_range_translation, TranslationType
import numpy as np
from generate_datasets.generators.input_image_generator import InputImagesGenerator

class TranslateGenerator(InputImagesGenerator):
    def __init__(self, translation_type, middle_empty,  jitter=0, **kwargs):
        """
        @param translation_type: could either be one TypeTranslation, or a dict of TypeTranslation (one for each class), or a tuple of two elements (x and y for the translated location) or a tuple of 4 elements (minX, maxX, minY, maxY)
        """
        super().__init__(**kwargs)

        self.translation_type = translation_type
        self.middle_empty = middle_empty
        self.jitter = jitter
        self.translations_range = {}
        self.p_boxes = {i: [1.0] for i in self.name_classes}

        def translation_type_to_str(translation_type):
            return str.lower(str.split(str(translation_type), '.')[1]) if isinstance(translation_type, TranslationType) else "_".join(str(translation_type).split(", "))

        # One key for each class
        if isinstance(self.translation_type, dict):
            self.translation_type_str = "".join(["".join([str(k), translation_type_to_str(v)]) for k, v in self.translation_type.items()])
            if len(self.translation_type_str) > 250:
                self.translation_type_str = 'multi_long'
            assert len(self.translation_type) == self.num_classes
            for idx, transl in self.translation_type.items():
                self.translations_range[idx] = self.fill_translation_range(transl, idx)

        # Same translation type for all classes
        # can be TranslationType, tuple (X, Y), tuple (minX, maxX, minY, maxY), or tuple of tuples
        if not isinstance(self.translation_type, dict):
            self.translation_type_str = translation_type_to_str(self.translation_type)
            for idx in self.name_classes:
                self.translations_range[idx] = self.fill_translation_range(self.translation_type, idx)

        self._finalize_init()

    def fill_translation_range(self, transl, idx):
        if isinstance(transl, TranslationType):
            return get_range_translation(transl,
                                         self.size_object[1],
                                         self.size_canvas,
                                         self.size_object[0],
                                         self.middle_empty,
                                         jitter=self.jitter)

        elif isinstance(transl, tuple) and len(np.array(transl).flatten()) == 2:
            return (transl[0],
                    transl[0] + 1,
                    transl[1],
                    transl[1] + 1)

        elif isinstance(transl, tuple) and len(transl) == 4:
            return transl

        elif isinstance(transl, tuple) and np.all([isinstance(transl[i], tuple) for i in range(len(transl))]):
            assert np.all([np.all([isinstance(t[i], tuple) for i in range(len(t))]) for t in self.translation_type.values()]), 'If one of the translation value is a box, they all need to be boxes (tuple of tuples)'
            self._get_translation = self._multi_area_translation
            area_boxes = np.array([(b[1] - b[0]) * (b[3] - b[2]) for b in transl])
            self.p_boxes[idx] = tuple(area_boxes / np.sum(area_boxes))
            return transl
        else:
            assert False, f'TranslationType type not understood: [{transl}]'

    def _get_translation(self, label, image_name=None, idx=None):
        return self._random_translation(label, image_name)

    def _multi_area_translation(selfclass_name, class_name, image_name=None, idx=None):
        b = np.random.choice(range(len(selfclass_name.translations_range[class_name])), p=selfclass_name.p_boxes[class_name])
        minX, maxX, minY, maxY = selfclass_name.translations_range[class_name][b]
        x = np.random.randint(minX, maxX)
        y = np.random.randint(minY, maxY)
        return x, y

    def _random_translation(self, class_name, image_name=None):
        minX, maxX, minY, maxY = self.translations_range[class_name]
        x = np.random.randint(minX, maxX)
        y = np.random.randint(minY, maxY)
        return x, y

