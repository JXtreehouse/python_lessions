"""The module for image pre-processing
This module include:
  Method:
    - lung_mask: Find the mask for lung in a ct image.
    - blob_detection: Detect blobs from MHD image.
    - search_candidates: It relies on the result of blob_detection.

Author: Jns Ridge--##--ridgejns@gmail.com
"""

import numpy as np
from src.imrp import mhd_io
from itertools import combinations
from skimage.feature import blob_dog, blob_log, blob_doh
from tqdm import tqdm
import cv2


def lung_mask(frames, low_area=300, high_area=10000, sym_coef=50, low_th=40, disp_progress=False):
    """Using image method to extract the mask of the lung(s)

    Args:
    frames: Slices of the ct image.
    low_area: Threshold of low area.
    high_area: Threshold of high area.
    sym_coef: Symmetry coefficient.
    low_th: Low threshold for dark area.
    disp_progress: Progress display flag.

    Returns:
    l_masks: Lung masks.
    l_convex_masks: Convex shape of lung masks.
    frames_contours_lung: Contours of lung masks.
    frames_contours_lung_convex: Contours of convex lung masks.
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
                        coef = (abs(centroids[comb[0]][0] + centroids[comb[1]][0] - frame.shape[1]) +
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
            elif len(contours_lung) == 1:
                # one large area was found, get the symmetric one of this lung.
                M = cv2.moments(contours_lung[0])
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroid_base = [cx, cy]

                # if (centroid_base[0] * 2 - frame.shape[1]) > (0.3 * sym_coef):
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


def blob_detection(mhd_img, masks, frames_contours, kernel_size=0, disp_progress=False):
    """Blob detection, using skimage's method

    Args:
    mhd_img: Image (class <MHD>).
    masks: Detection area in the image.
    frames_contours: Detection contours in the image.
    kernel_size: This parameter for expend the masks area.
    disp_progress: Progress display flag.

    Return:
    frames_blobs: Detected blobs in frames [[[z00, y00, x00, r00], [z01, y01, x01, r01]...]], [..., [], ...], ...].
    """
    frames = mhd_img.frames
    spacing = mhd_img.spacing
    r_pixels_min = max(1.3 / spacing[0], 1)
    r_pixels_max = 20 / spacing[0]
    r_pixels_max_bound = 5 / spacing[0]
    # min_s = max(round(r_pixels_min / 1.414 / 0.5) * 0.5, 1.5)
    # min_s = round(r_pixels_min / 1.414 / 0.5) * 0.5
    # max_s = round(r_pixels_max / 1.414 / 0.5) * 0.5
    min_s = round(r_pixels_min / 0.5) * 0.5
    max_s = round(r_pixels_max / 0.5) * 0.5
    # print(min_s, max_s)
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
            frame_blobs = blob_doh(frame_crop, min_sigma=1, max_sigma=max_s, overlap=0.1, threshold=0.005)
            # frame_blobs = blob_dog(frame_crop, min_sigma=min_s, max_sigma=max_s, sigma_ratio=1.6, threshold=0.1)
            # frame_blobs = blob_log(frame_crop, min_sigma=min_s, max_sigma=max_s, num_sigma=20, threshold=0.2,
            #                        overlap=0)
            # frame_blobs[:, 2] = 1.414 * frame_blobs[:, 2]

            # if i == 257:
            #     for blob in frame_blobs:
            #         cv2.circle(frame_crop, (int(blob[1]), int(blob[0])), int(blob[2] * 1.414), 255, 2)
            #     cv2.imshow('f', frame_crop)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
        except:
            frames_blobs.append([])
        else:
            if frame_blobs.shape[0] > 1:
                # if i == 257:
                #     cv2.imshow('cc',frame_crop)
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()
                # frame_blobs = np.add(frame_blobs, [y_min, x_min, 0])
                # frame_blobs = np.insert(frame_blobs, 0, [i], axis=1)
                # frames_blobs.append(frame_blobs)

                frame_blobs = np.add(frame_blobs, [y_min, x_min, 0])
                frame_blobs = frame_blobs.astype('int')
                rinsed_blobs = list()

                # get rid of the edge blobs
                for blob in frame_blobs:
                    # radius is too small
                    if (blob[2] < r_pixels_min) | (blob[2] > r_pixels_max):
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
                    _, blob_crop_th = cv2.threshold(blob_crop, 90, 255, cv2.THRESH_BINARY)
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
                            r_pixels_bound = min(blob[2], r_pixels_max_bound)
                            # r_pixels_bound = blob[2]
                            if dist > -r_pixels_bound:
                                # if dist > -10:
                                flag = 1
                                break
                    # flag = 1
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
    spacing: Image spacing.
    frames_blobs: Detected blobs in frames.

    Returns:
    cands: All connected areas [z, y, x, r, prev_idx, dist_prev, next_idx, dist_next].
    rinsed_cands: Result after rinse.
    """

    for i, frame_blobs in enumerate(frames_blobs):
        if len(frame_blobs) > 0:
            frames_blobs[i] = np.append(frame_blobs, -np.ones((frame_blobs.shape[0], 4)), 1)

    dif = 5
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
        # if len(cand) == 1:
        #     continue
        r_max = np.asarray(cand)[:, 3].max() * spacing[0]
        est_slices = int(np.max((np.round(r_max * 2 / spacing[2]), 1)))
        est_slices_range = (np.ceil(est_slices * 0.15).astype('int'),
                            np.ceil(est_slices * 1.05).astype('int'))
        # est_slices_ranges.append([est_slices_range, r_max])
        if (len(cand) < est_slices_range[0]) | (len(cand) > est_slices_range[1]):
            continue
        rinsed_cands.append(cand)
        cord = np.asarray(cand)[:, 0:2]
        cord_mean = cord.mean(0)
        rinsed_var.append(np.var(cord_mean - cord))

    return cands, rinsed_cands
