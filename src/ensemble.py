import argparse
import os
import sys

from predictors.ultralytics_predictor import UltralyticsPredictor
from predictors.rddc2020_predictor import Rddc2020Predictor
from predictors.megvii_predictor import MegviiPredictor
from ensemble_boxes import weighted_boxes_fusion

def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble model predictions.")
    parser.add_argument('--framework', type=str, required=True, choices=["yolov5", "yolov8", "yolov9", "yolox"], help='Framework to use.')
    parser.add_argument('--repo', type=str, required=True, choices=["ultralytics", "rddc2020", "megvii"], help='Repository to use.')
    parser.add_argument('--weights', type=str, required=True, help='Comma separated paths to weights.')
    parser.add_argument('--images', type=str, required=True, help='Path to test dataset directory or a txt file containing paths of all images.')
    parser.add_argument('--ensemble', type=str, required=True, choices=["wbf"], help='Ensemble algorithm to use.')
    parser.add_argument('--exp_file', type=str, required=False, help='Experiment description file i.e. model for YOLOX.')
    parser.add_argument('--conf', type=float, required=False, default=0.3, help='Confidence threshold for YOLOX.')
    parser.add_argument('--tsize', type=int, required=False, help='Test image size for YOLOX.')
    parser.add_argument('--iou', type=float, required=False, help='IOU Threshold')
    parser.add_argument('--agnostic-nms', action='store_true', help='Class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='Augmented inference')
    parser.set_defaults(
        weights='./predictors/yolov5/weights/IMSC/exp5_s_epoch25.pt,./predictors/yolov5/weights/IMSC/exp5_s_epoch50.pt,./predictors/yolov5/weights/IMSC/exp5_s_epoch75.pt,./predictors/yolov5/weights/IMSC/exp5_s_epoch100.pt',
        tsize=640,
        images='./data/test/smaller',
        conf=0.5,
        iou=0.999,
        framework='yolov5',
        repo='rddc2020',
        ensemble='wbf'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    # Print the parsed arguments
    print(f"Framework: {args.framework}")
    print(f"Repository: {args.repo}")
    print(f"Weights: {args.weights}")
    print(f"Images: {args.images}")
    print(f"Ensemble: {args.ensemble}")
    print(f"Experiment File: {args.exp_file}")
    print(f"Confidence Threshold: {args.conf}")
    print(f"Test Image Size: {args.tsize}")
    print(f"IOU Threshold: {args.iou}")
    print(f"Agnostic NMS: {args.agnostic_nms}")
    print(f"Augment: {args.augment}")
    
    weights = args.weights.split(',')
    exp_files = args.exp_file.split(',') if args.exp_file else []
    images_path = args.images

    # Create predictor object based on the repository
    if args.repo == "ultralytics":
        predictor = UltralyticsPredictor(args.framework)
    elif args.repo == "megvii":
        predictor = MegviiPredictor(args.framework, exp_files, args.conf, args.tsize)
    elif args.repo == "rddc2020":
        predictor = Rddc2020Predictor(args.framework, conf_thres=args.conf, iou_thres=args.iou, img_size=args.tsize, agnostic_nms=args.agnostic_nms, augment=args.augment)
    else:
        raise NotImplementedError()

    predictor.load(weights, images_path)
    predictions = predictor.predict()

    boxes_list, scores_list, labels_list = predictions

    ensembled_boxes = []
    ensembled_scores = []
    ensembled_labels = []

    weights = [1] * len(boxes_list)  # Adjust weights as needed
    iou_thr = 0.5
    skip_box_thr = 0.0001


    for i in range(len(predictor.images)):
        image_boxes_list = [model_boxes[i] for model_boxes in boxes_list]
        image_scores_list = [model_scores[i] for model_scores in scores_list]
        image_labels_list = [model_labels[i] for model_labels in labels_list]

        boxes, scores, labels = weighted_boxes_fusion(
            image_boxes_list, image_scores_list, image_labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr
        )
        
        ensembled_boxes.append(boxes)
        ensembled_scores.append(scores)
        ensembled_labels.append(labels)

    with open('./results/ensembled_results.csv', 'w') as f:
        f.write("image_id, detection_details\n")
        for img_idx, img_path in enumerate(predictor.images):
            img_name = os.path.basename(img_path)
            img_width, img_height = predictor.get_image_size(img_path)
            result_str = ' '.join([
                f"{int(label)} {int(box[0])} {int(box[1])} {int(box[2])} {int(box[3])}"
                for box, score, label in zip(
                    [predictor.denormalize_box(b, img_width, img_height) for b in ensembled_boxes[img_idx]], 
                    ensembled_scores[img_idx], ensembled_labels[img_idx]
                )
            ])
            f.write(f"{img_name}, {result_str}\n")

if __name__ == "__main__":
    """
    example command:for running the rddc2020 predictor - 
    python .\ensemble.py --weights .\predictors\yolov5\weights\IMSC\exp5_s_epoch25.pt,.\predictors\yolov5\weights\IMSC\exp5_s_epoch50.pt,.\predictors\yolov5\weights\IMSC\exp5_s_epoch75.pt,.\predictors\yolov5\weights\IMSC\exp5_s_epoch100.pt --tsize 640 --images .\data\test\smaller --conf 0.5 --iou 0.999 --agnostic-nms --augment --framework yolov5 --repo rddc2020 --ensemble wbf
    pip install ensemble-boxes, ultralytics, opencv, numpy, pandas
    """ 
    main()
