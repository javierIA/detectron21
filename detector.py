import cv2
from importlib_metadata import metadata 
import numpy as np
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog,Metadata
from detectron2.utils.visualizer import ColorMode


class Detector:
    def __init__(self, model_type):
        self.cfg = get_cfg()
        if(model_type=="Detection"):
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        elif(model_type=="PanopticSegmentation"):
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
        elif(model_type=="Keypoint"):
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        elif(model_type=="InstanceSegmentation"):
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        elif(model_type=="LVIS"):
            self.cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")
        elif(model_type=="CUSTOM"):
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = "models/modelos.pth"
            self.cfg.MODE.NUM_CLASSES = 2
        
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        
        self.cfg.MODEL.DEVICE = "cpu"
        self.predictor = DefaultPredictor(self.cfg)
    

    def OnImage(self, image):
        image = cv2.imread(image)
        predictions = self.predictor(image)
        print(predictions["instances"].pred_classes)
        print(predictions["instances"].pred_boxes)
        print(predictions["instances"].scores)
        #print(predictions["instances"].pred_masks)
        #metadata = Metadata(thing_classes=["pothole"])
        viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2,instance_mode=
        ColorMode.IMAGE_BW)

        out = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
        cv2.imshow("Image",out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
    
    def OnWebcam(self, webcam):
        cap = cv2.VideoCapture(int(webcam))
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                predictions = self.predictor(frame)
                viz = Visualizer(frame[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2,instance_mode=
                ColorMode.IMAGE_BW)

                out = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
                cv2.imshow("Image",out.get_image()[:, :, ::-1])
                cv2.waitKey(1)
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

    def OnVideo(self, video):
        cap = cv2.VideoCapture(video)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                predictions = self.predictor(frame)
                viz = Visualizer(frame[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2,instance_mode=
                ColorMode.IMAGE_BW)
                out = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
                cv2.imshow("Image",out.get_image()[:, :, ::-1])
                cv2.waitKey(1)
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
    
