"""
This script is used to split the train and validation files from coco annotations.
The train and validation splits are saved in different json files in the specified folder.
"""
import argparse
import json
import os.path
import pandas as pd

from sklearn.model_selection import train_test_split


# Path: utils/train_val_split_annotations.py
def train_val_split_annotations(
    src_annotations_path: str,
    train_split: float = 0.8,
    val_split: float = 0.2,
    seed: int = 42,
    output_dir: str = None,
):
    """
    This function is used to split the train and validation files from coco annotations.
    The train and validation splits are saved in different json files in the specified folder.

    Arguments:
        src_annotations_path {str} -- Path to the coco annotations file
        train_split {float} -- Percentage of the train split
        val_split {float} -- Percentage of the validation split
        seed {int} -- Seed for the random state
        output_dir {str} -- Path to the output directory
    """
    with open(src_annotations_path, "r") as f:
        coco = json.load(f)
    images = coco["images"]
    info = coco["info"]
    categories = coco["categories"]
    annotations = coco["annotations"]

    # create a dataframe from the annotations
    annotations_df = pd.DataFrame(annotations)
    # create a new column with the image id
    annotations_df["image_id"] = annotations_df["image_id"]
    # create a new column with the category id
    annotations_df["category_id"] = annotations_df["category_id"]

    # create images dataframe
    images_df = pd.DataFrame(images)
    # create a new column with the image id
    images_df["id"] = images_df["id"]

    # split the images dataframe into train and validation
    train_images_df, val_images_df = train_test_split(
        images_df, train_size=train_split, random_state=seed
    )

    # filter the annotations dataframe for test and val images
    train_annotations_df = annotations_df[
        annotations_df["image_id"].isin(train_images_df["id"])
    ]
    val_annotations_df = annotations_df[
        annotations_df["image_id"].isin(val_images_df["id"])
    ]

    # create train and val annotations
    train_annotations = {
        "info": info,
        "categories": categories,
        "images": train_images_df.to_dict("records"),
        "annotations": train_annotations_df.to_dict("records"),
    }
    val_annotations = {
        "info": info,
        "categories": categories,
        "images": val_images_df.to_dict("records"),
        "annotations": val_annotations_df.to_dict("records"),
    }

    # Save the annotations in a JSON file
    if output_dir is None:
        output_dir = os.path.dirname(src_annotations_path)
    train_annotations_path = os.path.join(
        output_dir,
        os.path.basename(src_annotations_path).replace(".json", "_train.json"),
    )
    val_annotations_path = os.path.join(
        output_dir, os.path.basename(src_annotations_path).replace(".json", "_val.json")
    )
    with open(train_annotations_path, "w") as f:
        json.dump(train_annotations, f)
    with open(val_annotations_path, "w") as f:
        json.dump(val_annotations, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split the train and validation files from coco annotations"
    )
    parser.add_argument(
        "--src_annotations_path",
        type=str,
        help="Path to the coco annotations file",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Percentage of the train split",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Percentage of the validation split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the random state",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to the output directory",
    )
    args = parser.parse_args()
    train_val_split_annotations(
        args.src_annotations_path,
        args.train_split,
        args.val_split,
        args.seed,
        args.output_dir,
    )
