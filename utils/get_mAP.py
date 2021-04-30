"""
 Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
"""


def voc_ap(rec, prec):
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


"""
mAP Calculator

pred_data:
{image1: [{'box': [left, top, right, bottom], 'score': score, 'label': label}, ...]}
gt_data:
{image1: [{'box': [left, top, right, bottom], 'label': label}, ...]}

If only for face detection,
then label='face'

"""


def get_mAP(pred_data, gt_data, overlap_threshold=0.5):
    # format gt data
    gt_counter_per_class = {}
    out_gt_data = {}
    for image_id in gt_data.keys():
        out_gt_data[image_id] = []
        for item in gt_data[image_id]:
            label = item['label']
            out_gt_data[image_id].append({'box': item['box'], 'label': label, 'used': False})
            # count that object
            if label in gt_counter_per_class:
                gt_counter_per_class[label] += 1
            else:
                # if class didn't exist yet
                gt_counter_per_class[label] = 1

    gt_data = out_gt_data
    gt_classes = list(gt_counter_per_class.keys())
    # let's sort the classes alphabetically
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)

    # format pred data
    detect_results = {}
    for class_index, label in enumerate(gt_classes):
        bounding_boxes = []
        for image_id in pred_data.keys():
            for item in pred_data[image_id]:
                if item['label'] == label:
                    bounding_boxes.append({'box': item['box'], 'image_id': image_id, 'score': item['score']})
        # sort detection-results by decreasing score
        bounding_boxes.sort(key=lambda x: float(x['score']), reverse=True)
        detect_results[label] = bounding_boxes

    sum_AP = 0.0

    count_true_positives = {}
    for class_index, label in enumerate(gt_classes):
        count_true_positives[label] = 0
        dr_data = detect_results[label]

        """
         Assign detection-results to ground-truth objects
        """
        nd = len(dr_data)
        tp = [0] * nd  # creates an array of zeros of size nd
        fp = [0] * nd
        for idx, detection in enumerate(dr_data):
            image_id = detection['image_id']
            ground_truth_data = gt_data[image_id]
            ovmax = -1
            gt_match = -1
            # load detected object bounding-box
            bb = detection['box']
            bb = [float(x) for x in bb]
            for obj in ground_truth_data:
                # look for a label match
                if obj['label'] == label:
                    bbgt = obj['box']
                    bbgt = [float(x) for x in bbgt]
                    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0] + 1) * (
                                bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj

            # set minimum overlap
            min_overlap = overlap_threshold

            if ovmax >= min_overlap:
                if not bool(gt_match["used"]):
                    # true positive
                    tp[idx] = 1
                    gt_match["used"] = True
                    count_true_positives[label] += 1
                else:
                    # false positive (multiple detection)
                    fp[idx] = 1
            else:
                # false positive
                fp[idx] = 1

        # print(tp)
        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        # print(tp)
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[label]
        # print(rec)
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        # print(prec)

        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_AP += ap

    mAP = sum_AP / n_classes

    return mAP
