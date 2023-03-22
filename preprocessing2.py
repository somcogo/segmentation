import glob
import os
import time

import h5py
import numpy as np

# f_alt = h5py.File('../data/segmentation/alt.hdf5', 'w')

# f_alt.create_dataset('img', shape=(172, 64, 64, 64), chunks=(1, 64, 64, 64), dtype='float32')
# alt_img_ds = f_alt['img']
# f_alt.create_dataset('mask', shape=(172, 64, 64, 64), chunks=(1, 64, 64, 64), dtype='float32')
# alt_mask_ds = f_alt['mask']

# alt_path_list = glob.glob('../data/segmentation/alt/*')
# alt_id_list = []
# for i, path in enumerate(alt_path_list):
#     t1 = time.time()
#     id = os.path.split(path)[-1][:7]
#     alt_id_list.append(id)

#     file = h5py.File('../data/segmentation/alt/{}.hdf5'.format(id), 'r')
#     img = np.array(file['img'][:])
#     mask = np.array(file['mask'][:])

#     center_coord = file.attrs['center_coords']
#     center_coord = center_coord.astype(int)
#     side_length = 64 // 2
#     img = img[
#         center_coord[0] - side_length: center_coord[0] + side_length,
#         center_coord[1] - side_length: center_coord[1] + side_length,
#         center_coord[2] - side_length: center_coord[2] + side_length
#     ]
#     mask = mask[
#         center_coord[0] - side_length: center_coord[0] + side_length,
#         center_coord[1] - side_length: center_coord[1] + side_length,
#         center_coord[2] - side_length: center_coord[2] + side_length
#     ]

#     alt_img_ds[i] = img
#     alt_mask_ds[i] = mask
#     f_alt.flush()
#     t2 = time.time()
#     print('{}/172 {} {}'.format(i+1, id, t2-t1))

# f_alt.create_dataset('patient_id', data=alt_id_list)

f_neu = h5py.File('../data/segmentation/neu.hdf5', 'w')

f_neu.create_dataset('img', shape=(143, 64, 64, 64), chunks=(1, 64, 64, 64), dtype='float32')
neu_img_ds = f_neu['img']
f_neu.create_dataset('mask', shape=(143, 64, 64, 64), chunks=(1, 64, 64, 64), dtype='float32')
neu_mask_ds = f_neu['mask']

neu_path_list = glob.glob('../data/segmentation/neu/*')
neu_id_list = []
for i, path in enumerate(neu_path_list):
    t1 = time.time()
    id = os.path.split(path)[-1][:7]
    neu_id_list.append(id)

    file = h5py.File('../data/segmentation/neu/{}.hdf5'.format(id), 'r')
    img = np.array(file['img'][:])
    mask = np.array(file['mask'][:])
    # file.close()

    center_coord = file.attrs['center_coords']
    center_coord = center_coord.astype(int)
    side_length = 64 // 2
    img = img[
        center_coord[0] - side_length: center_coord[0] + side_length,
        center_coord[1] - side_length: center_coord[1] + side_length,
        center_coord[2] - side_length: center_coord[2] + side_length
    ]
    mask = mask[
        center_coord[0] - side_length: center_coord[0] + side_length,
        center_coord[1] - side_length: center_coord[1] + side_length,
        center_coord[2] - side_length: center_coord[2] + side_length
    ]

    neu_img_ds[i] = img
    neu_mask_ds[i] = mask
    # f_neu.flush()
    t2 = time.time()
    print('{}/143 {} {}'.format(i+1, id, t2-t1))

f_neu.create_dataset('patient_id', data=neu_id_list)