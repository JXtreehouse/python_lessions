"""The module for mhd image io interface
This module include:
  Class:
    - MHD: Structure of medical image
    - NODE: Structure of node information
  Method:
    - mhd_normalize: Normalization for mhd image
    - im_convert_type: Image convert the type
    - resample: MHD image re-sampling, <create a same scale for difference image>
    - read: MHD image reading
    - write: MHD image writing
    - node_crop: Cropping the node (node info is necessary)

Author: Jns Ridge--##--ridgejns@gmail.com
"""

import SimpleITK as sitk
import numpy as np
from scipy import ndimage
import warnings
import copy


# Hounsfield Unit
# HU = {'Air': -1000, 'Lung': -500, 'Fat': [-100, -50], 'Water': [0], 'CSF': 15, 'Kidney': 30,
#       'Blood': [35, 45], 'Muscle': [10, 40], 'GreyMatter': [37, 45], 'WhiteMatter': [20, 30],
#       'Liver': [40, 60], 'SoftTissue': [100, 300], 'Bone': [700, 300]}


class MHD(object):
    """structure of medical image"""

    def __init__(self):
        self.height = 0
        self.width = 0
        self.depth = 0
        self.offset = None
        self.spacing = None
        self.frames = None
        self.itype = 'origin'  # image data type: origin, uint8, uint16, float
        # self.dtype = 'origin'
        # self.itk_img = None

    pass


class NODE(object):
    """structure of node information"""

    def __init__(self):
        self.coord_x = 0
        self.coord_y = 0
        self.coord_z = 0
        self.diameter = 0
        self.ctype = 'world'  # coordinate type: world,  pixel

    pass


def im_normalize(frames, min_bound=-1000.0, max_bound=400.0):
    """Normalization of ct image

    Args:
    frames: Slices of the ct image.
    min_bound:
    max_bound:

    Return:
    frames: Normalized image.
    """

    if not isinstance(frames, np.ndarray):
        raise ValueError('frames type %s is invalid, it must be %s' % (type(frames), np.ndarray))
    frames = np.divide((frames - min_bound), (max_bound - min_bound))
    frames[frames > 1] = 1.
    frames[frames < 0] = 0.
    frames = frames.astype('float32')
    return frames


def mhd_convert_type(mhd_img, itype='uint8'):
    """Convert the type of the image

    Args:
    mhd_img: Image (class <MHD>).
    dtype: Output data type.

    Return:
    frames: Converted image.
    """
    # Warning: image convert will take irreversible precision loss.

    if not isinstance(mhd_img, MHD):
        raise ValueError('frames type %s is invalid, it must be %s' % (type(mhd_img), MHD))
    result = copy.deepcopy(mhd_img)
    frames = result.frames
    if (frames.max() > 1) | (frames.min() < 0):
        frames = im_normalize(frames)
    if itype.lower() == 'uint8':
        frames = np.rint(frames * 255).astype('uint8')
        result.itype = 'uint8'
    elif itype.lower() == 'uint16':
        frames = np.rint(frames * 65535).astype('uint16')
        result.itype = 'uint16'
    elif itype.lower() == 'float':
        result.itype = 'float'
        pass
    else:
        raise ValueError('dtype <%s> is invalid.' % itype)
    result.frames = frames
    return result


def resample(image, spacing, new_spacing=(1, 1, 1)):
    """Resample the image stack

    Args:
    image: Original image (3D array, same with the frames).
    spacing: Original image spacing, [scale_x, scale_y, scale_z].
    new_spacing: Resample spacing.

    Returns:
    image: Re-sampled image.
    new_spacing: Real new_spacing.
    """

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * np.flip(resize_factor, 0)
    new_shape = np.round(new_real_shape)
    real_resize_factor = np.flip(new_shape / image.shape, 0)
    new_spacing = spacing / real_resize_factor
    image = ndimage.interpolation.zoom(image, np.flip(real_resize_factor, 0), mode='nearest')
    return image, new_spacing


def read(mhd_path, itype='origin'):
    """Read mhd image

    Args:
    mhd_path: CT image path.

    Return:
    result: Wrapped image data (class <MHD>).
    """

    result = MHD()
    itk_img = sitk.ReadImage(mhd_path)
    result.frames = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (notice the ordering)
    result.depth, result.height, result.width = result.frames.shape  # heightXwidth constitute the transverse plane
    result.offset = np.array(itk_img.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
    result.spacing = np.array(itk_img.GetSpacing())  # spacing of voxels in world coor. (mm)
    result.itype = itype.lower()
    # result.frames[result.frames < -1024] = -1024
    if result.itype != 'origin':
        result = mhd_convert_type(result, result.itype)
    return result


def write(mhd_path, img, offset=None, spacing=None, compress=False):
    """Write mhd image

    Args:
    mhd_path: Output file path.
    img: CT image (frames, MHD or sitkImage).

    Return:
    True/False.
    """

    if isinstance(img, sitk.SimpleITK.Image):  # img type is itk_img, write the image directly
        itk_img = img
    elif isinstance(img, np.ndarray):  # img type is array, transform the array to itk_img and write
        itk_img = sitk.GetImageFromArray(img)
        if offset is not None:
            itk_img.SetOrigin(offset)
        if spacing is not None:
            itk_img.SetSpacing(spacing)
    elif isinstance(img, MHD):  # img type is MHD, extract the array from MHD and transform and write
        itk_img = sitk.GetImageFromArray(img.frames)
        if offset is not None:
            warnings.warn('param <offset> is invalid, when img type is %s' % MHD, RuntimeWarning)
        if spacing is not None:
            warnings.warn('param <spacing> is invalid, when img type is %s' % MHD, RuntimeWarning)
        if img.offset is not None:
            itk_img.SetOrigin(img.offset)
        if img.spacing is not None:
            itk_img.SetSpacing(img.spacing)
    else:
        raise ValueError('img type %s is invalid, it must be %s, %s or %s' %
                         (type(img), sitk.SimpleITK.Image, np.ndarray, MHD))
    try:
        sitk.WriteImage(itk_img, mhd_path, compress)
        return True
    except:
        return False


def node_crop(mhd_img, node_info, resample_spacing=None, crop_shape=(48, 48, 48)):
    """Cropping the node to crop_shape s.t. resample_spacing

    Args:
    mhd_img: CT image (MHD).
    node_info: Info of node (NODE).
    resample_spacing: Tuple (spacing_x, spacing_y, spacing_z).
    crop_shape: Shape of the output image (x, y, z).

    Return:
    cropped Image(MHD).
    """

    if not isinstance(mhd_img, MHD):
        raise ValueError('mhd_img type %s is invalid, it must be %s.' % (type(mhd_img), MHD))
    if not isinstance(node_info, NODE):
        raise ValueError('node_info type %s is invalid, it must be %s.' % (type(node_info), NODE))

    if node_info.ctype.lower() == 'world':
        real_x = (node_info.coord_x - mhd_img.offset[0]) / mhd_img.spacing[0]
        real_y = (node_info.coord_y - mhd_img.offset[1]) / mhd_img.spacing[1]
        real_z = (node_info.coord_z - mhd_img.offset[2]) / mhd_img.spacing[2]
    elif node_info.ctype.lower() == 'pixel':
        real_x, real_y, real_z = node_info.coord_x, node_info.coord_y, node_info.coord_z
    else:
        raise ValueError('node\'s coordinate type <%s> is invalid' % node_info.ctype.lower())

    if resample_spacing is not None:
        cs2 = np.ceil(np.divide(crop_shape, mhd_img.spacing) * resample_spacing).astype('int')
    else:
        # cs2 = np.ceil(np.divide(crop_shape, mhd_img.spacing)).astype('int')
        cs2 = np.asarray(crop_shape, 'int')

    # this cbox is expended, it will be cropped before return.
    cbox = np.array([[real_x - cs2[0] / 2 - 1, real_x + cs2[0] / 2 + 1],
                     [real_y - cs2[1] / 2 - 1, real_y + cs2[1] / 2 + 1],
                     [real_z - cs2[2] / 2 - 1, real_z + cs2[2] / 2 + 1]]).astype('int')
    bound = np.array([[0, mhd_img.width], [0, mhd_img.height], [0, mhd_img.depth]])
    if (cbox[:, 0] < bound[:, 0]).sum() | (cbox[:, 1] > bound[:, 1]).sum():
        return None

    cropped_frames = mhd_img.frames[cbox[2][0]:cbox[2][1], cbox[1][0]:cbox[1][1], cbox[0][0]:cbox[0][1]].copy()

    node_img = MHD()

    if resample_spacing:
        node_img.frames, node_img.spacing = resample(cropped_frames, mhd_img.spacing, resample_spacing)
    else:
        node_img.frames = cropped_frames
        node_img.spacing = mhd_img.spacing

    node_img.width, node_img.height, node_img.depth = crop_shape
    node_img.offset = (0, 0, 0)
    c_depth, c_height, c_width = node_img.frames.shape
    ah_depth = int((c_depth - node_img.depth) / 2)
    ah_height = int((c_height - node_img.height) / 2)
    ah_width = int((c_width - node_img.width) / 2)
    cbox = np.array([[ah_width, crop_shape[0] + ah_width],
                     [ah_height, crop_shape[1] + ah_height],
                     [ah_depth, crop_shape[2] + ah_depth]], 'int')
    node_img.frames = node_img.frames[cbox[2][0]:cbox[2][1], cbox[1][0]:cbox[1][1], cbox[0][0]:cbox[0][1]]
    if node_img.frames.shape != (node_img.depth, node_img.height, node_img.width):
        print('catch one')
        return None
    # node_img.itk_img = sitk.GetImageFromArray(node_img.frames)
    # node_img.itk_img.SetSpacing(resample_spacing)
    return node_img
