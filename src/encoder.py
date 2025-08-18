import os
import cv2
from pathlib import Path

import numpy as np

def get_small_pokemon(array):
    import numpy as np
    new_array = np.array(array)[:,1:]
    rows_to_keep = ~(new_array == '~').all(axis=1)
    cols_to_keep = ~(new_array == '~').all(axis=0)
    new_array = new_array[rows_to_keep][:, cols_to_keep]
    if new_array.shape[0]>22 or new_array.shape[1]>22:
        new_array = []
    else:
        padded = np.full((22, 22), "~", dtype=str)
        row_size,col_size = new_array.shape
        row_start = (22 - row_size) // 2
        col_start = (22 - col_size) // 2
        row_end = row_start+row_size
        col_end = col_start+col_size
        padded[row_start:row_end, col_start:col_end] = new_array
        row_numbers = np.array([f"{i:02}" for i in range(22)]).reshape(22, 1)
        padded_with_row_nums = np.hstack((row_numbers, padded))
        new_array = padded_with_row_nums.tolist()
    return new_array


class Pokedex:

    @staticmethod
    def encode(image):
        """ """

        width, height, _ = image.shape

        array = []
        for y in range(0, height, 2):

            row = ["%02d" % (y // 2)]
            for x in range(0, width, 2):
                r, g, b = image[y, x] // 64
                is_blank = min(image[y, x]) > 245 or max(image[y, x]) < 10
                char = (
                    "~"
                    if is_blank
                    else chr(r * 4**2 + g * 4**1 + b * 4**0 + 59)
                )

                row.append(char)
            array.append(row[:-1])

        return array

    @staticmethod
    def decode(array):
        """ """

        array = [[ord(pixel) - 59 for pixel in row] for row in array]

        def idx_to_rgb(pixel):
            r = (pixel % 16) * 64
            g = ((pixel % 16) // 4) * 64
            b = ((pixel % 16) % 4) * 64
            return [r, g, b]

        array = [[idx_to_rgb for pixel in row] for row in array]

        return np.array(array)

    @staticmethod
    def array_to_text(array):
        return "\n".join([" ".join(r) for i, r in enumerate(array)])

    @classmethod
    def batch_files_encoding(cls, images_dir, dataset_dir):
        """ """

        paths = [str(x) for x in Path(images_dir).glob("**/*.png")]

        for img_name in paths:

            image = cv2.imread(img_name)
            array = cls.encode(image)
            array = get_small_pokemon(array)

            if not array:
                continue

            batch = cls._augmentation(array)

            for name, array in batch.items():
                file_name = (img_name
                    .replace(".png", f"_{name}.txt")
                    .replace("pokemons/", "pokemons_txt/")
                )
                with open(file_name, "w") as file:
                    text = cls.array_to_text(array)
                    text = text.replace("\n"," ") #.replace(" ","")
                    text = "[BOS] " + text + " [EOS]"
                    file.write(text)

    @classmethod
    def _augmentation(cls, array):
        """ """

        batch = {}
        batch["original"] = array
        batch["fliped"] = [row[0:1] + row[:0:-1] for row in array]
        # TODO color transformation

        return batch