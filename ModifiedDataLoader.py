import torch
from torch.utils.data import Dataset
import numpy as np
import zipfile
from io import BytesIO
import json
import skimage as sk  # for sk.measure.label
import tifffile
import matplotlib as mlp


class LiveCellImageDataset(Dataset):
    def __init__(self,
                 wells=['B3', 'B4', 'B5', 'B7', 'B8', 'B9', 'E3', 'E4', 'E5'],
                 pixel_width=600,
                 phase=True, red=False, tcell_mask=False, cancer_mask=False,
                 fpath='', startFrame=150, numFrames=50):
        self.fpath = fpath
        self.f_end_name = '_start_0_end_350_nuc_15_cyto_75.zip'
        self.wells = wells
        self.n_wells = len(wells)
        self.pixel_width = pixel_width
        self.n_per_side = 600 - self.pixel_width + 1
        self.phase = phase
        self.red = red
        self.tcell_mask = tcell_mask
        self.cancer_mask = cancer_mask
        self.start_frame = startFrame
        self.num_frames = numFrames

    def __len__(self):
        return self.n_wells * self.n_per_side ** 2 * self.num_frames  # Adjust the multiplier if you have more frames
        # (second one)

    def __getitem__(self, idx):
        frames_per_well = self.n_per_side ** 2 * self.num_frames  # Adjust if you have more frames per well
        well = int(np.floor(idx / frames_per_well))
        well_name = self.wells[well]
        frame_idx_within_well = idx % frames_per_well
        frame_within_sequence = frame_idx_within_well // (self.n_per_side ** 2)
        frame_position = frame_idx_within_well % (self.n_per_side ** 2)

        firstFrameNumber = self.start_frame + frame_within_sequence  # Adjust this if you start from a different frame

        fname = self.fpath + 'cart_' + well_name + self.f_end_name
        dcl_ob = load_data_local(fname)

        x0 = int(frame_position / self.n_per_side)
        xf = x0 + self.pixel_width
        y0 = int(frame_position % self.n_per_side)
        yf = y0 + self.pixel_width

        if well_name in ['B3', 'B4', 'B5']:
            label = 0
        elif well_name in ['B7', 'B8', 'B9']:
            label = 1
        elif well_name in ['E3', 'E4', 'E5']:
            label = 2
        else:
            label = -1  # should never actually be used; -1 means testing, not training

        image_phase = torch.tensor(dcl_ob['X'][0, firstFrameNumber, x0:xf, 0, y0:yf])
        image_red = torch.tensor(dcl_ob['X'][1, firstFrameNumber, x0:xf, 0, y0:yf])
        image = torch.stack([image_phase, image_red, image_phase], dim=0)  # Create an RGB-like image

        return image, label, well_name


def load_data_local(filepath):
    f = zipfile.ZipFile(filepath, 'r')
    file_bytes = f.read("cells.json")
    with BytesIO() as b:
        b.write(file_bytes)
        b.seek(0)
        cells = json.load(b)
    file_bytes = f.read("divisions.json")
    with BytesIO() as b:
        b.write(file_bytes)
        b.seek(0)
        divisions = json.load(b)
    file_bytes = f.read("X.ome.tiff")
    with BytesIO() as b:
        b.write(file_bytes)
        b.seek(0)
        X = sk.io.imread(b, plugin="tifffile")
    file_bytes = f.read("y.ome.tiff")
    with BytesIO() as b:
        b.write(file_bytes)
        b.seek(0)
        y = sk.io.imread(b, plugin="tifffile")
    dcl_ob = {
        'X': np.expand_dims(X, 3),
        'y': np.expand_dims(y, 3),
        'divisions': divisions,
        'cells': cells}
    return dcl_ob


debug = True

# class TCellDataset(Dataset,
#                    wells = ['B3', 'B7', 'E3'],
#                    pixel_width = 600,
#                    phase = True, red = True, tcell_mask = True, red_mask = True):
#     def __init__(self):
#         #something
#         x = 0

#     def __len__(self):
#         return 0

#     def __getitem__(self, idx):
#         return 0
