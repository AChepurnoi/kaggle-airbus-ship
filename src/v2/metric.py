import numpy as np
from sklearn.metrics import fbeta_score

def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
            metric.append(0)
            continue
        if np.count_nonzero(t) >= 1 and np.count_nonzero(p) == 0:
            metric.append(0)
            continue
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
            metric.append(1)
            continue

        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = np.sum(intersection > 0) / np.sum(union > 0)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)

def iou(img_true, img_pred):
    i = np.sum((img_true * img_pred) > 0)
    u = np.sum((img_true + img_pred) > 0) + 0.0000000000000000001  # avoid division by zero
    return i / u


thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def f2(masks_true, masks_pred):
    # a correct prediction on no ships in image would have F2 of zero (according to formula),
    # but should be rewarded as 1
    if np.sum(masks_true) == np.sum(masks_pred) == 0:
        return 1.0

    f2_total = 0
    ious = {}
    for t in thresholds:
        tp, fp, fn = 0, 0, 0
        for i, mt in enumerate(masks_true):
            found_match = False
            for j, mp in enumerate(masks_pred):
                key = 100 * i + j
                if key in ious.keys():
                    miou = ious[key]
                else:
                    miou = iou(mt, mp)
                    ious[key] = miou  # save for later
                if miou >= t:
                    found_match = True
            if not found_match:
                fn += 1

        for j, mp in enumerate(masks_pred):
            found_match = False
            for i, mt in enumerate(masks_true):
                miou = ious[100 * i + j]
                if miou >= t:
                    found_match = True
                    break
            if found_match:
                tp += 1
            else:
                fp += 1
        f2 = (5 * tp) / (5 * tp + 4 * fn + fp)
        f2_total += f2

    return f2_total / len(thresholds)
