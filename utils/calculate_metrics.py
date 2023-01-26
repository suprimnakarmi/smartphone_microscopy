import json
import pandas as pd
from torchvision.ops import box_iou
import pandas as pd
import json
import torch
from os import PathLike


def calculate_precision_recall_f1(
    pred_annotations_file: PathLike,
    gt_annotations_file: PathLike,
    conf_threshold: float = 0,
):
    """
    This function calculates the precision, recall and f1 score for each class
    in the given ground truth and predicted annotations.

    Arguments:
        pred_annotations_file: path to the predicted annotations file
        gt_annotations_file: path to the ground truth annotations file
        conf_threshold: confidence threshold to filter out predictions

    Returns:
        metrics_df: dataframe containing the precision, recall and f1 score for each class
        precisions: dictionary containing the precision for each class
        recalls: dictionary containing the recall for each class
        confidence_scores: dictionary containing the confidence score for each class

    """
    # Precision x Recall is obtained individually by each class
    # Loop through each class and calculate the precision and recall

    # Precision = TP / (TP + FP)
    # Recall = TP / (TP + FN)
    with open(gt_annotations_file) as f:
        gt_annotations = json.load(f)

    with open(pred_annotations_file) as f:
        pred_annotations = json.load(f)

    gt_annotations_df = pd.DataFrame(gt_annotations["annotations"])
    pred_annotations_df = pd.DataFrame(pred_annotations)

    # change bbox width and height to x2, y2
    pred_annotations_df["bbox"] = pred_annotations_df["bbox"].apply(
        lambda x: [x[0], x[1], x[0] + x[2], x[1] + x[3]]
    )
    gt_annotations_df["bbox"] = gt_annotations_df["bbox"].apply(
        lambda x: [x[0], x[1], x[0] + x[2], x[1] + x[3]]
    )

    categories = sorted(gt_annotations_df.category_id.unique())

    # dataframe to store the precision, recall and f1 score for each class
    metrics_df = pd.DataFrame(
        columns=["category", "precision", "recall", "f1_score", "TP", "FP"]
    )

    precisions = dict((category, []) for category in categories)
    recalls = dict((category, []) for category in categories)
    confidence_scores = dict((category, []) for category in categories)

    for category in categories:
        # get the ground truth annotations for the current class
        gt_annotations_df_class = gt_annotations_df[
            gt_annotations_df.category_id == category
        ]
        # get the predicted annotations for the current class
        pred_annotations_df_class = pred_annotations_df[
            pred_annotations_df.category_id == category
        ]

        # sort the predicted annotations by score
        pred_annotations_df_class = pred_annotations_df_class.sort_values(
            by="score", ascending=False
        )

        # filter predictions with score > conf_threshold
        if conf_threshold:
            pred_annotations_df_class = pred_annotations_df_class[
                pred_annotations_df_class.score > conf_threshold
            ]

        true_positives_class = 0
        false_positives_class = 0

        # get image ids for the current class from both ground truth and predicted annotations
        image_ids = pred_annotations_df_class["image_id"].unique()
        images_len = len(image_ids)

        # get the confidence scores of the annotations with image id in image_ids
        confidence_scores[category] = list(
            pred_annotations_df_class[
                pred_annotations_df_class.image_id.isin(image_ids)
            ]["score"].values
        )

        for image in image_ids:
            # get the ground truth annotations for the current image
            gt_annotations_df_image = gt_annotations_df_class[
                gt_annotations_df_class.image_id == image
            ]
            # get the predicted annotations for the current image
            pred_annotations_df_image = pred_annotations_df_class[
                pred_annotations_df_class.image_id == image
            ]

            # get the ground truth bounding boxes
            gt_bboxes = list(gt_annotations_df_image.bbox.values)
            gt_bboxes = torch.tensor(gt_bboxes)

            # get the predicted bounding boxes
            pred_bboxes = list(pred_annotations_df_image.bbox.values)
            pred_bboxes = torch.tensor(pred_bboxes)

            if len(gt_bboxes) == 0:
                # false_positives_class += len(pred_bboxes)
                for i in range(len(pred_bboxes)):
                    false_positives_class += 1
                    precisions[category].append(
                        true_positives_class
                        / (true_positives_class + false_positives_class)
                    )
                    recalls[category].append(
                        true_positives_class / gt_annotations_df_class.shape[0]
                    )
                continue

            # get the intersection over union for each predicted bounding box
            ious = box_iou(pred_bboxes, gt_bboxes)

            # get the maximum iou for each ground truth bounding box
            max_ious, _ = torch.max(ious, dim=0)

            # get the indices of the predicted bounding boxes with iou > 0.5
            tp_indices = torch.where(max_ious >= 0.5)[0]
            # print(ious)

            # get the indices of the predicted bounding boxes with iou < 0.5
            fp_indices = torch.where(max_ious < 0.5)[0]

            for i in range(len(tp_indices)):
                true_positives_class += 1
                precisions[category].append(
                    true_positives_class
                    / (true_positives_class + false_positives_class)
                )
                recalls[category].append(
                    true_positives_class / gt_annotations_df_class.shape[0]
                )

            for i in range(len(fp_indices)):
                false_positives_class += 1
                precisions[category].append(
                    true_positives_class
                    / (true_positives_class + false_positives_class)
                )
                recalls[category].append(
                    true_positives_class / gt_annotations_df_class.shape[0]
                )

        # calculate the precision and recall
        precision = true_positives_class / (
            true_positives_class + false_positives_class
        )
        recall = true_positives_class / gt_annotations_df_class.shape[0]

        category_name = gt_annotations["categories"][category]["name"]
        f1_score = 2 * (precision * recall) / (precision + recall)

        category_metrics_df = pd.DataFrame(
            {
                "category": category_name,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "TP": true_positives_class,
                "FP": false_positives_class,
            },
            index=[0],
        )

        # concatenate the metrics for the current class to the metrics dataframe
        metrics_df = pd.concat([metrics_df, category_metrics_df], axis=0)

    return metrics_df, precisions, recalls, confidence_scores
