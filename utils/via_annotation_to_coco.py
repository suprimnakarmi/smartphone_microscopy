"""
Script to convert the annotations from VIA format to COCO format.
The annotations are saved in a JSON file.
"""
import math
import os
from argparse import ArgumentParser
import mmcv

from detectron2.structures import BoxMode


# function to rotate point
def rotate_point(cx, cy, x, y, angle):
    """Rotate a point (x, y) around the point (cx, cy) by the given angle (in radians)"""
    return (
        cx + (x - cx) * math.cos(angle) - (y - cy) * math.sin(angle),
        cy + (x - cx) * math.sin(angle) + (y - cy) * math.cos(angle),
    )


def ellipse_to_bounding_box(ellipse):
    """
    Convert ellipse, which is a dictionary with keys cx, cy, rx, ry,
    and theta in radians, into a bounding box that encloses the ellipse.
    """
    cx = ellipse["cx"]
    cy = ellipse["cy"]
    rx = ellipse["rx"]
    ry = ellipse["ry"]
    theta = ellipse["theta"]
    orientation = math.radians(theta)
    top_left_x = cx - rx / 2
    top_left_y = cy - ry / 2
    bottom_right_x = cx + rx / 2
    bottom_right_y = cy + ry / 2

    # rotate the corner points around the center point
    top_left_x, top_left_y = rotate_point(cx, cy, top_left_x, top_left_y, orientation)
    bottom_right_x, bottom_right_y = rotate_point(
        cx, cy, bottom_right_x, bottom_right_y, orientation
    )

    width = bottom_right_x - top_left_x
    height = bottom_right_y - top_left_y
    return [
        top_left_x,
        top_left_y,
        width,
        height,
    ]


def cysts_annotations_to_coco(img_dir, input_json_file, out_json_file):
    # Function to change the annotations from VIA format to COCO format

    imgs_anns = mmcv.load(input_json_file)

    imgs_anns_coco = []
    images = []
    obj_count = 0
    # Loop through the entries in the JSON file
    for idx, v in enumerate(mmcv.track_iter_progress(imgs_anns.values())):
        filename = os.path.join(img_dir, v["filename"])
        if os.path.isfile(filename):
            pass
        else:
            print("Image not found", filename)
            continue
        height, width = mmcv.imread(filename).shape[:2]
        images.append(dict(id=idx, file_name=filename, height=height, width=width))

        # one image can have multiple annotations, therefore this loop is needed
        for data_anno in v["regions"]:
            # reformat the polygon information to fit the specifications
            anno = data_anno["shape_attributes"]
            cx = anno["cx"]
            cy = anno["cy"]
            rx = anno["rx"]
            ry = anno["ry"]
            theta = anno["theta"]
            region_attributes = data_anno["region_attributes"]["Cyst"]

            # specify the category_id to match with the class.
            if "Giardia" in region_attributes:
                category_id = 1
            elif "Crypto" in region_attributes:
                category_id = 0

            # calculate the bounding box's corner points
            top_left_x, top_left_y, width, height = ellipse_to_bounding_box(anno)

            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=category_id,
                bbox=[
                    top_left_x,
                    top_left_y,
                    width,
                    height,
                ],
                segmentation=[],
                iscrowd=0,
                area=width * height,
            )
            imgs_anns_coco.append(data_anno)
            obj_count += 1
    # save the annotations in a JSON file
    coco_format_json = dict(
        info={
            "description": f"Cysts annotations in COCO format for {input_json_file.split('/')[-1].rsplit('.', 1)[0]}"
        },
        categories=[{"id": 0, "name": "Crypto"}, {"id": 1, "name": "Giardia"}],
        images=images,
        annotations=imgs_anns_coco,
    )
    mmcv.dump(coco_format_json, out_json_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_dir", type=str, help="Path to the base directory")
    parser.add_argument(
        "--img_dir",
        type=str,
        help="Path to the directory, relative to the base directory, containing the images",
    )
    parser.add_argument(
        "--input_json_file",
        type=str,
        help="Path to the JSON file, relative to the base directory, containing the annotations in VIA format",
    )
    parser.add_argument(
        "--out_json_file",
        type=str,
        help="Path to the JSON file, relative to the base directory, to save the annotations in COCO format",
    )
    args = parser.parse_args()
    args.img_dir = os.path.join(args.base_dir, args.img_dir)
    args.input_json_file = os.path.join(args.base_dir, args.input_json_file)
    args.out_json_file = os.path.join(args.base_dir, args.out_json_file)
    cysts_annotations_to_coco(args.img_dir, args.input_json_file, args.out_json_file)
