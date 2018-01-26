"""Some utilities
This module include:
  Class:
    - MHDSet:

  Method:
    - anim_play:

Author: Jns Ridge--##--ridgejns@gmail.com
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


class MHDSet(object):
    """This is a class for search the mhd image path from a folder.

    Class:
      - __MHDSetData: structure

    Method:
      - search_mhd_set: search the path of the mhd image
      - destroy_image_set: destroy

    """

    class __MHDSetData(object):
        """MHD Image set data structure."""

        def __init__(self):
            self.img_location = list()
            self.description = None
            self.count = 0

        def __repr__(self):
            if self.count == 0:
                s = "It is empty."
            else:
                s = 'Folder <%s> includes <%d> image(s).' % (self.description, self.count)
            return s

    def __init__(self):
        self.image_set = None
        self.valid_folder_count = 0
        self.__img_suffixes = {'.mhd'}

    def search_mhd_set(self, location, recursive=False):
        """Search mhd image set

        Args:
        location: folder path
        recursive: True/False

        """
        if not os.path.isdir(location):
            return
        description = str(os.path.split(location)[-1])
        if description.startswith('.'):
            return
        imgsd = self.__MHDSetData()
        imgsd.description = description
        __f_list = os.listdir(location)

        if recursive is True:
            for f in __f_list:
                f_ap = os.path.join(location, f)
                if os.path.isdir(f_ap):
                    self.search_mhd_set(f_ap, recursive=True)
                elif os.path.splitext(f_ap)[-1].lower() in self.__img_suffixes:
                    imgsd.count += 1
                    imgsd.img_location.append(f_ap)
            if imgsd.count > 0:
                if self.image_set is None:
                    self.image_set = list()
                self.valid_folder_count += 1
                self.image_set.append(imgsd)
        else:
            for f in __f_list:
                f_ap = os.path.join(location, f)
                if os.path.splitext(f_ap)[-1].lower() in self.__img_suffixes:
                    imgsd.count += 1
                    imgsd.img_location.append(f_ap)
            if imgsd.count > 0:
                if self.image_set is None:
                    self.image_set = list()
                self.valid_folder_count += 1
                self.image_set.append(imgsd)

    def destroy_image_set(self):
        """Destroy image set

        """
        self.image_set = None
        self.valid_folder_count = 0


class FileSet(object):
    """This is a class for search some files with specific postfix.

    Class:
      - __FileSetData: structure

    Method:
      - search_file_set: search the path of the data
      - destroy_file_set: destroy

    """

    class __FileSetData(object):
        """File set data structure."""

        def __init__(self):
            self.location = list()
            self.description = None
            self.count = 0

        def __repr__(self):
            if self.count == 0:
                s = 'Folder <%s> is empty.' % self.description
            else:
                s = 'Folder <%s> includes <%d> file(s).' % (self.description, self.count)
            return s

    def __init__(self, postfix):
        self.file_set = None
        self.valid_folder_count = 0
        for idx, pfx in enumerate(postfix):
            if pfx[0] != '.':
                postfix[idx] = '.%s' % pfx
        self.__file_postfix = set(postfix)

    def search_file_set(self, location, recursive=False):
        """Search file set

        Args:
        location: Folder path
        recursive: True/False

        """
        if not os.path.isdir(location):
            return
        description = str(os.path.split(location)[-1])
        if description.startswith('.'):
            return
        fsd = self.__FileSetData()
        fsd.description = description
        __f_list = os.listdir(location)

        if recursive is True:
            for f in __f_list:
                f_ap = os.path.join(location, f)
                if os.path.isdir(f_ap):
                    self.search_file_set(f_ap, recursive=True)
                elif os.path.splitext(f_ap)[-1].lower() in self.__file_postfix:
                    fsd.count += 1
                    fsd.location.append(f_ap)
            if fsd.count > 0:
                if self.file_set is None:
                    self.file_set = list()
                self.valid_folder_count += 1
                self.file_set.append(fsd)
        else:
            for f in __f_list:
                f_ap = os.path.join(location, f)
                if os.path.splitext(f_ap)[-1].lower() in self.__file_postfix:
                    fsd.count += 1
                    fsd.location.append(f_ap)
            if fsd.count > 0:
                if self.file_set is None:
                    self.file_set = list()
                self.valid_folder_count += 1
                self.file_set.append(fsd)

    def destroy_file_set(self):
        """Destroy file set

        """
        self.file_set = None
        self.valid_folder_count = 0


def anim_play(ct_imgs, output=None):
    depth = len(ct_imgs)
    fig = plt.figure()

    init = plt.imshow(ct_imgs[0], cmap=plt.cm.bone)

    def animate(i):
        init.set_array(ct_imgs[i])
        return init,

    anim = animation.FuncAnimation(fig, animate, frames=range(depth), interval=50, blit=True)
    if output is not None:
        anim.save(output, writer="ffmpeg", fps=24)
    plt.show()
