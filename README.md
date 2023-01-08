# Talk2Car-Trajectory

This is the dataset that accompanies the paper [Talk2Car: Predicting Physical Trajectories for
Natural Language Commands](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9961196) accepted in IEEE Access.

Talk2Car-Trajectory is an extension to [Talk2Car](https://github.com/talk2car/Talk2Car) which is built on [nuScenes](https://www.nuscenes.org/).

# Annotation format
Each json from the dataset is a dictionary where the key is the command token and the value is a dictionary of the following format.

```
{
 "image": "img name",
 "top-down": "top down image name"
 "command": "given command"
 "destinations": [[x,y]], #is a list of (x, y) pairs where each pair is a destination in the top-down image
 "trajectories": [[(x0,y0), (xn, yn)]], #is a list of lists of (x, y) pairs where each pair is a point in the trajectory in the top-down image
 "egobbox_top": [ 4 x 2 list], # contains the corners of the ego vehicle bounding box in the top-down image.
 "all_detections_top": [64 x 4 x 2 list], # contains the corners of all detected objects in the top-down image.
 "detected_object_classes": [64 list], # contains the class of each detected object.
 "all_detections_front": [64 x 4 x 2 list], # contains the corners of all detected objects in the frontal image.
 "predicted_referred_obj_index":  [64 list], # contains the index of the predicted referred object.
 "detection_scores":  [64 list], # contains the confidence score of each detected object.
 "gt_referred_obj_top": [4 x 2 list], # contains the corners of the ground truth referred object in the top-down image (only in train and val).
 "gt_referred_obj_front": [x0, y0, x1, y1], # contains the corners of the ground truth referred object in the frontal image (only in train and val).
}              
```

***Note:*** The data in `trajectories` contains the nodes of the trajectories. 
However, all these trajectories may have varying number of nodes. To resolve this issue, we apply spline interpolation to all trajectories to make them have the same number of nodes.
The code for this spline interpolation can be found in the utils folder.
Additionally, we provide a visualization script `visualize.py` to visualize the trajectories and the referred object in the top-down image and frontal images.

# How to use

1. Download top-down images [here](https://drive.google.com/file/d/1lrgghIVYPxCboZ77eTO8cdFcm_6mcZga/view?usp=sharing) and put the images in the data folder.
2. Download the frontal images [here](https://drive.google.com/file/d/1bhcdej7IFj5GqfvXGrHGPk2Knxe77pek/view?usp=sharing) and put the images in the data folder.
3. Download the Talk2Car-Trajectory dataset [here](https://drive.google.com/file/d/1IPsgQknSWCFGqgR0EaWNB8fDcyyohMXL/view?usp=share_link) and put all files in the data folder. We also include pre-extracted commmand embeddings with a Sentence-BERT model in the .h5 files in this zip.
4. Run `visualize.py` to visualize a sample of the dataset

# Integration with Talk2Car

Drag the Talk2Car-Trajectory dataset into the `data/commands` folder of Talk2Car.
Next, when calling the `get_talk2car_class`, set `load_talk2car_trajectory` to `True`.
Talk2Car-Trajectory will now be loaded.

# Citation
If you use this dataset, please consider using the following citation:

```
@article{deruyttere2022talk2car,
  title={Talk2Car: Predicting physical trajectories for natural language commands},
  author={Deruyttere, Thierry and Grujicic, Dusan and Blaschko, Matthew B and Moens, Marie-Francine},
  journal={Ieee Access},
  volume={10},
  pages={123809--123834},
  year={2022},
  publisher={IEEE}
}
```