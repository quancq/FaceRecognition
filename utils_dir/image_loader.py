from utils_dir import utils
import numpy as np
from imgaug import augmenters as iaa
import random
import os
import cv2


def generate_batch(dataset_dir, batch_size=64, image_size=160):

    seq = iaa.Sequential([
        iaa.Scale({"height": image_size, "width": image_size}),
        iaa.Fliplr(0.5, random_state=7),
        iaa.Affine(scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, rotate=(-15, 15))
    ])

    random.seed(7)
    mid_names = utils.get_dir_names(dataset_dir)
    random.shuffle(mid_names)

    map_mid_idx = {mid: 0 for mid in mid_names}
    map_mid_fpaths = {mid: utils.get_file_paths(os.path.join(dataset_dir, mid))
                      for mid in mid_names}

    mid_idx = 0
    max_iter = 0
    while True:
        x_batch1, x_batch2, y_batch = [], [], []
        for i in range(int(batch_size / 2)):
            mid_name = mid_names[mid_idx]
            # Generate same class pair
            fpaths = map_mid_fpaths.get(mid_name)
            path_idx = map_mid_idx.get(mid_name)

            img1 = cv2.imread(fpaths[path_idx])
            # seq.show_grid(img1, cols=8, rows=8)
            img2 = cv2.imread(fpaths[(path_idx + 1) % len(fpaths)])

            x_batch1.append(seq.augment_image(img1))
            x_batch2.append(seq.augment_image(img2))
            y_batch.append(1)

            path_idx = (path_idx + 1) % len(fpaths)
            map_mid_idx.update({mid_name: path_idx})
            if path_idx == 0:
                random.shuffle(fpaths)

            # Generate different class pair
            next_mid_name = mid_names[(mid_idx + 1) % len(mid_names)]
            fpaths = map_mid_fpaths.get(next_mid_name)
            img2 = cv2.imread(random.choice(fpaths))

            x_batch1.append(seq.augment_image(img1))
            x_batch2.append(seq.augment_image(img2))
            y_batch.append(0)

            mid_idx = (mid_idx + 1) % len(mid_names)
            if mid_idx == 0:
                random.shuffle(mid_names)

        # yield np.array(x_batch), np.array(y_batch)
        yield [x_batch1, x_batch2], y_batch
        # max_iter += 1
        # if max_iter > 4:
        #     break


def test_gen_batch():
    dataset_dir = "../Dataset/Test_Split/Train"
    batch_size = 4

    for x_batch, y_batch in generate_batch(dataset_dir, batch_size):
        # print(x_batch.shape, y_batch.shape)
        break


if __name__ == "__main__":
    test_gen_batch()
    pass
