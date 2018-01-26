"""The module for image pre-processing
This module include:
  Method:
    - lung_mask: find the mask for lung in a ct image.
    - blob_detection:
    - search_candidates:

Author: Jns Ridge--##--ridgejns@gmail.com
"""

import numpy as np
from src.imrp import mhd_io
from itertools import combinations
from skimage.feature import blob_dog, blob_log, blob_doh
from tqdm import tqdm
import cv2


def lung_mask(frames, low_area=300, high_area=20000, sym_coef=50, low_th=40, disp_progress=False):
    """Using image method to extract the mask of the lung(s)

    Args:
    frames: slices of the ct image
    low_area: threshold of low area
    high_area: threshold of high area
    sym_coef: symmetry coefficient
    low_th: low threshold for dark area
    disp_progress: progress display flag

    Returns:
    l_masks: lung masks
    l_convex_masks: convex shape of lung masks
    frames_contours_lung: contours of lung masks
    frames_contours_lung_convex: contours of convex lung masks

    """

    if not isinstance(frames, np.ndarray):
        raise ValueError('frames type %s is invalid, it must be %s' % (type(frames), np.ndarray))

    if frames.dtype != 'uint8':
        frames = mhd_io.im_convert_type(frames, dtype='uint8')

    l_masks = np.zeros(frames.shape, 'uint8')
    l_convex_masks = np.zeros(frames.shape, 'uint8')
    frames_contours_lung = []
    frames_contours_lung_convex = []

    if disp_progress is True:
        pbar = tqdm(desc='masking', total=frames.shape[0])
    else:
        pbar = None

    for i, frame in enumerate(frames):
        try:
            pbar.update(1)
        except:
            pass

        th, th_frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, contours, hierarchy = cv2.findContours(th_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # sorted contours index (top to bottom)
        scs_idx = sorted(range(len(contours)), key=lambda k: cv2.contourArea(contours[k]), reverse=True)
        hierarchy = hierarchy[0]

        pt_contours_lung = []  # potential contours of lung
        pt_contours_lung_convex = []

        pt_contour_idx_lung = hierarchy[scs_idx[0]][2]  # potential contour index of lung
        while 1:
            if pt_contour_idx_lung == -1:
                break
            pt_contours_lung.append(contours[pt_contour_idx_lung])
            pt_contours_lung_convex.append(cv2.convexHull(contours[pt_contour_idx_lung]))
            pt_contour_idx_lung = hierarchy[pt_contour_idx_lung][0]

        pt_areas_lung = []  # potential areas of lung
        pt_areas_lung_convex = []

        for contour in pt_contours_lung:
            pt_areas_lung.append(cv2.contourArea(contour))
        for contour in pt_contours_lung_convex:
            pt_areas_lung_convex.append(cv2.contourArea(contour))

        pt_scs_idx = sorted(range(len(pt_areas_lung)), key=lambda k: pt_areas_lung[k], reverse=True)

        for idx in pt_scs_idx[-1::-1]:
            if pt_areas_lung[idx] > low_area:
                break
            pt_scs_idx.pop()

        contours_lung = []
        contours_lung_convex = []

        if len(pt_scs_idx) >= 1:
            # exist potential contours of lung
            pt_scs_idx_tmp = pt_scs_idx.copy()
            for idx in pt_scs_idx_tmp:
                # get the area larger than <high_area> firstly.
                if pt_areas_lung_convex[idx] > high_area:
                    contours_lung.append(pt_contours_lung[idx])
                    contours_lung_convex.append(pt_contours_lung_convex[idx])
                    pt_scs_idx.remove(idx)

            if len(contours_lung) < 1:
                # large area not found
                if len(pt_scs_idx) >= 2:
                    # search the symmetric area first.
                    centroids = []
                    for idx in pt_scs_idx:
                        # contours_lung_convex.append(pt_contours_lung_convex[idx])
                        M = cv2.moments(pt_contours_lung[idx])
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        centroids.append([cx, cy])
                    combs = list(combinations(range(len(pt_scs_idx)), 2))
                    coefs = []
                    for comb in combs:
                        coef = (abs(centroids[comb[0]][0] + centroids[comb[1]][0] - 512) +
                                abs(centroids[comb[0]][1] - centroids[comb[1]][1]))
                        coefs.append(coef)
                    # print(i, 'sys', coefs)
                    c_idx = np.argmin(coefs)
                    if coefs[c_idx] < sym_coef:
                        idx0 = pt_scs_idx[combs[c_idx][0]]
                        idx1 = pt_scs_idx[combs[c_idx][1]]
                        contours_lung.append(pt_contours_lung[idx0])
                        contours_lung.append(pt_contours_lung[idx1])
                        contours_lung_convex.append(pt_contours_lung_convex[idx0])
                        contours_lung_convex.append(pt_contours_lung_convex[idx1])
                        pt_scs_idx.remove(idx0)
                        pt_scs_idx.remove(idx1)

                        # if len(contours_lung) < 1:
                        #     # using threshold to find.
                        #     pt_scs_idx_tmp = pt_scs_idx.copy()
                        #     low_th_tmp = low_th
                        #     idx0 = -1
                        #     for idx in pt_scs_idx_tmp:
                        #         # find the contours, which avg_gray large than threshold
                        #         mask_tmp = np.zeros(frame.shape, 'uint8')
                        #         cv2.drawContours(mask_tmp, [pt_contours_lung[idx]], -1, 255, cv2.FILLED)
                        #         avg_gray, _, _, _ = cv2.mean(frame, mask_tmp)
                        #         # print(i + 1, 'avg_gray', avg_gray)
                        #
                        #         if avg_gray > low_th_tmp:
                        #             contours_lung.append(pt_contours_lung[idx])
                        #             contours_lung_convex.append(pt_contours_lung_convex[idx])
                        #             pt_scs_idx.remove(idx)

                        #     if avg_gray > low_th_tmp:
                        #         low_th_tmp = avg_gray
                        #         idx0 = idx
                        # print(i + 1, low_th_tmp)
                        # if idx0 != -1:
                        #     contours_lung.append(pt_contours_lung[idx0])
                        #     contours_lung_convex.append(pt_contours_lung_convex[idx0])
                        #     pt_scs_idx.remove(idx0)

            elif len(contours_lung) == 1:
                # one large area was found, get the symmetric one of this lung.
                M = cv2.moments(contours_lung[0])
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroid_base = [cx, cy]

                if (centroid_base[0] * 2 - frame.shape[1]) > (0.3 * sym_coef):
                    if len(pt_scs_idx) >= 1:
                        centroids = []
                        for idx in pt_scs_idx:
                            M = cv2.moments(pt_contours_lung[idx])
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                            centroids.append([cx, cy])
                        coefs = []
                        for centroid in centroids:
                            coef = (0.5 * abs(centroid[0] + centroid_base[0] - frame.shape[1]) +
                                    0.5 * abs(centroid[1] - centroid_base[1]))
                            coefs.append(coef)
                        c_idx = np.argmin(coefs)
                        # print(i + 1, 'large_sys', coefs)
                        if coefs[c_idx] < sym_coef * 1.1:
                            contours_lung.append(pt_contours_lung[pt_scs_idx[c_idx]])
                            contours_lung_convex.append(pt_contours_lung_convex[pt_scs_idx[c_idx]])
                        else:
                            contours_lung.pop()
                            contours_lung_convex.pop()
            else:
                pass
            cv2.drawContours(l_masks[i], contours_lung, -1, 255, cv2.FILLED)
            cv2.drawContours(l_convex_masks[i], contours_lung_convex, -1, 255, cv2.FILLED)
        frames_contours_lung.append(contours_lung)
        frames_contours_lung_convex.append(contours_lung_convex)
    try:
        pbar.close()
    except:
        pass
    return l_masks, l_convex_masks, frames_contours_lung, frames_contours_lung_convex


def blob_detection(frames, spacing, masks, frames_contours, method='doh', kernel_size=0, disp_progress=False):
    """Blob detection, using skimage's method

    Args:
    frames: slices of the ct image
    masks: detection area in the image
    frames_contours: detection contours in the image
    method: detection method (doh, log, dog)
    kernel_size: this parameter for expend the masks area
    disp_progress: progress display flag

    Return:
    frames_blobs: detected blobs in frames [[[z00, y00, x00, r00], [z01, y01, x01, r01]...]], [..., [], ...], ...]

    """
    r_pixels_min = max(1.2 / spacing[0], 1)
    # r_pixels_min = 0
    if kernel_size > 0:
        # expend the size of the mask
        kernel = np.ones((kernel_size, kernel_size), 'uint8')
        for i, mask in enumerate(masks):
            if masks[i].sum() == 0:
                continue
            masks[i] = cv2.dilate(mask, kernel, iterations=1)
            # masks[i] = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    masks = masks.astype('bool')

    # define the method of the blob detection, default using <doh>
    if method.lower() == 'doh':
        detectObj = blob_doh
    elif method.lower() == 'dog':
        detectObj = blob_dog
    elif method.lower() == 'log':
        detectObj = blob_log
    else:
        raise ValueError('invalid method <%s>' % method)

    frames_blobs = list()
    if disp_progress is True:
        pbar = tqdm(desc='blob detecting', total=frames.shape[0])
    else:
        pbar = None
    frames_copy = frames.copy()
    for i, frame in enumerate(frames_copy):
        try:
            pbar.update(1)
        except:
            pass
        contours = frames_contours[i]
        if len(contours) == 0:
            # frame[::] = 0
            frames_blobs.append([])
            continue
        x_min, x_max, y_min, y_max = frame.shape[1], 0, frame.shape[0], 0
        for contour in contours:
            x_min_tmp = contour[:, :, 0].min()
            x_max_tmp = contour[:, :, 0].max()
            y_min_tmp = contour[:, :, 1].min()
            y_max_tmp = contour[:, :, 1].max()
            if x_min > x_min_tmp:
                x_min = x_min_tmp
            if x_max < x_max_tmp:
                x_max = x_max_tmp
            if y_min > y_min_tmp:
                y_min = y_min_tmp
            if y_max < y_max_tmp:
                y_max = y_max_tmp
        mask = masks[i]
        frame[~mask] = 0
        frame_crop = frame[y_min:y_max, x_min:x_max]
        # frame_blobs = detectObj(frame_crop, min_sigma=1, max_sigma=25, threshold=.1)
        try:
            frame_blobs = detectObj(frame_crop, min_sigma=1, max_sigma=20)
        except:
            frames_blobs.append([])
        else:
            if frame_blobs.shape[0] > 1:
                frame_blobs = np.add(frame_blobs, [y_min, x_min, 0])
                frame_blobs = frame_blobs.astype('int')
                rinsed_blobs = list()

                # get rid of the edge blobs
                for blob in frame_blobs:
                    # radius is too small
                    if blob[2] <= r_pixels_min:
                        continue
                    x_min = blob[1] - blob[2]
                    x_max = blob[1] + blob[2] + 1
                    y_min = blob[0] - blob[2]
                    y_max = blob[0] + blob[2] + 1
                    # out of boundary
                    # print(x_min, y_min, x_max, y_max)
                    if (x_min < 0) | (y_min < 0) | (x_max > frames.shape[2]) | (y_max > frames.shape[1]):
                        continue
                    blob_crop = np.copy(frame[y_min:y_max, x_min:x_max])
                    _, blob_crop_th = cv2.threshold(blob_crop, 140, 255, cv2.THRESH_BINARY)
                    blob_mask = np.zeros(blob_crop_th.shape, 'uint8')
                    blob_mask = cv2.circle(blob_mask, (blob[2], blob[2]), blob[2] - 1, 255, cv2.FILLED)
                    blob_mask = blob_mask.astype('bool')
                    white_sum = (blob_crop_th[blob_mask] == 255).sum()
                    black_sum = (blob_crop_th[blob_mask] == 0).sum() + 0.1
                    wb_rate = white_sum / black_sum
                    if wb_rate < 0.6:
                        continue

                    flag = 0
                    if flag == 0:
                        for contour in contours:
                            dist = cv2.pointPolygonTest(contour, (blob[1], blob[0]), True)
                            if dist > -blob[2]:
                                # if dist > -10:
                                flag = 1
                                break

                    if flag == 1:
                        rinsed_blobs.append(blob)
                rinsed_blobs = np.asanyarray(rinsed_blobs)
                if rinsed_blobs.shape[0] == 0:
                    frames_blobs.append([])
                else:
                    frame_blobs = np.insert(rinsed_blobs, 0, [i], axis=1)
                    # sequence of the blob [z, y, x, r]
                    frames_blobs.append(frame_blobs)
            else:
                frames_blobs.append([])
    try:
        pbar.close()
    except:
        pass
    return frames_blobs


def search_candidates(spacing, frames_blobs):
    """Search the linked area (candidate) in m3d image

    Args:
    spacing: image spacing
    frames_blobs: detected blobs in frames
    disp_progress:

    Returns:
    cands: all connected areas [z, y, x, r, prev_idx, dist_prev, next_idx, dist_next]
    rinsed_cands: result after rinse
    """

    for i, frame_blobs in enumerate(frames_blobs):
        if len(frame_blobs) > 0:
            frames_blobs[i] = np.append(frame_blobs, -np.ones((frame_blobs.shape[0], 4)), 1)

    dif = 3
    frames_blobs_less = frames_blobs.copy()
    frames_blobs_less.pop()
    for i, frame_blobs in enumerate(frames_blobs_less):
        if len(frame_blobs) == 0:
            continue
        for j, blob in enumerate(frame_blobs):
            next_frame_blobs = frames_blobs[i + 1]
            for k, blob_next in enumerate(next_frame_blobs):
                dist = np.linalg.norm([blob[1] - blob_next[1], blob[2] - blob_next[2]])
                if (dist < blob[3]) & (abs(blob[0] - blob_next[0]) < dif) & ((blob[7] == -1) | (dist < blob[7])):
                    blob[6] = k
                    blob[7] = dist
                    blob_next[4] = j
                    blob_next[5] = dist

    cands = list()
    for i, frame_blobs in enumerate(frames_blobs_less):
        if len(frame_blobs) == 0:
            continue
        for j, blob in enumerate(frame_blobs):
            blob_tmp = blob
            if blob_tmp[4] == -1:
                cand = list()
                cand.append(blob_tmp)
                while blob_tmp[6] != -1:
                    blob_next = frames_blobs[int(blob_tmp[0]) + 1][int(blob_tmp[6])]
                    blob_tmp = blob_next
                    cand.append(blob_tmp)
                cands.append(cand)

    rinsed_cands = list()
    rinsed_var = list()
    # est_slices_ranges = list()
    for cand in cands:
        if len(cand) == 1:
            continue
        r_max = np.asarray(cand)[:, 3].max() * spacing[0]
        est_slices = int(np.max((np.round(r_max * 2 / spacing[2]), 1)))
        est_slices_range = (np.ceil(est_slices * 0.2).astype('int'),
                            np.ceil(est_slices * 1.05).astype('int'))
        # est_slices_ranges.append([est_slices_range, r_max])
        if (len(cand) < est_slices_range[0]) | (len(cand) > est_slices_range[1]):
            continue
        rinsed_cands.append(cand)
        cord = np.asarray(cand)[:, 0:2]
        cord_mean = cord.mean(0)
        rinsed_var.append(np.var(cord_mean - cord))

    return cands, rinsed_cands
