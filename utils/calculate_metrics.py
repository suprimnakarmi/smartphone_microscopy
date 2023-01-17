import json
import pandas as pd
from torchvision.ops import box_iou
import pandas as pd
import json
import torch


def calculate_precision_recall_f1(pred_annotations_file, gt_annotations_file):
    # Precision x Recall is obtained individually by each class
    # Loop through each class and calculate the precision and recall

    # Precision = TP / (TP + FP)
    # Recall = TP / (TP + FN)
    gt_annotations = json.load(open(gt_annotations_file))
    pred_annotations = json.load(open(pred_annotations_file))

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
        columns=["category_id", "precision", "recall", "f1_score", "TP", "FP"]
    )

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

        # filter predictions with score > 0.3
        pred_annotations_df_class = pred_annotations_df_class[
            pred_annotations_df_class.score > 0.3
        ]

        true_positives_class = 0
        false_positives_class = 0

        # get image ids for the current class from both ground truth and predicted annotations
        image_ids = pred_annotations_df_class["image_id"].unique()
        images_len = len(image_ids)

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

            # only take the predicted bounding boxes which have a score > 0.3
            pred_bboxes = list(pred_annotations_df_image.bbox.values)
            pred_bboxes = torch.tensor(pred_bboxes)

            if len(gt_bboxes) == 0:
                false_positives_class += len(pred_bboxes)
                continue

            # get the intersection over union for each predicted bounding box
            ious = box_iou(gt_bboxes, pred_bboxes)

            # get the maximum iou for each ground truth bounding box
            max_ious, _ = torch.max(ious, dim=0)

            # get the indices of the predicted bounding boxes with iou > 0.5
            tp_indices = torch.where(max_ious >= 0.5)[0]
            # print(ious)

            # get the indices of the predicted bounding boxes with iou < 0.5
            fp_indices = torch.where(max_ious < 0.5)[0]

            # update the true positives and false positives
            true_positives_class += len(tp_indices)
            false_positives_class += len(fp_indices)

        # print actual number of ground truth annotations and predicted annotations for the current class
        print(
            "Actual:",
            gt_annotations_df_class.shape[0],
            "Predicted:",
            pred_annotations_df_class.shape[0],
        )

        # print true positives and false positives for the current class
        print(
            "True positives:",
            true_positives_class,
            "False positives:",
            false_positives_class,
        )
        # calculate the precision and recall
        precision = true_positives_class / (
            true_positives_class + false_positives_class
        )
        recall = true_positives_class / gt_annotations_df_class.shape[0]

        category_name = gt_annotations["categories"][category]["name"]
        print(f"Category: {category_name}, Precision: {precision}, Recall: {recall}")
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(f"F1 Score: {f1_score}")

        metrics_df = metrics_df.append(
            {
                "category_id": category,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "TP": true_positives_class,
                "FP": false_positives_class,
            },
            ignore_index=True,
        )

    return metrics_df
