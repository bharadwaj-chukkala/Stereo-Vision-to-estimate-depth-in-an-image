# Stereo Vision to estimate depth in an image

## Project Description

In this project, we are going to implement the concept of Stereo Vision. We will be given 3 different datasets, each of them contains 2 images of the same scenario but taken from two different camera angles. By comparing the information about a scene from 2 vantage points, we can obtain the 3D information by examining the relative positions of objects.

### Objective

To find the depth in an image by following the stereo vision pipeline.

## Dependencies

* [Matplotlib](https://matplotlib.org/) `pip install matplotlib`
* [OpenCV](https://opencv.org/) `pip install opencv-python`
* [NumPy](https://numpy.org/) `pip install numpy`
* [tqdm](https://tqdm.github.io/) `pip install tqdm`
* *The dataset used for this project is MiddleBury Stereo Dataset*

## Contents

```
├───data
│   ├───curule
│   ├───octagon
│   └───pendulum
├───bchukkal_proj3_report.pdf
├───stereo_vision.py
└───results
```

## Instructions for Usage

1. Clone the repository

```
git clone https://github.com/bharadwaj-chukkala/Stereo-Vision-to-estimate-depth-in-an-image.git
```

2. Install Python 3.9 and the libraries mentinoned below prior to running the code
3. Go to the root directory from your IDE.
4. Please mention the path to the datasets wherever necessary.
5. Run the `stereo_vision.py` file as it is.
6. Note: if dataset and results are not given, please paste the py file in the folder where dataset is present and also create a results folder in the directory where you run the code.

## Results

|                                    | Curule                                                                                                                                          | Octagon                                                                                                                                     | Pendulum                                                                                                                                    |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Epipolar geometry<br />(Rectified) | ![1671441533984](https://github.com/bharadwaj-chukkala/Stereo-Vision-to-estimate-depth-in-an-image/blob/master/results/epi_polar_lines_1.png)     | ![1671441808456](https://github.com/bharadwaj-chukkala/Stereo-Vision-to-estimate-depth-in-an-image/blob/master/results/epi_polar_lines_2.png) | ![1671441813305](https://github.com/bharadwaj-chukkala/Stereo-Vision-to-estimate-depth-in-an-image/blob/master/results/epi_polar_lines_3.png) |
| Disparity<br />(Heat)              | ![1671441938738](https://github.com/bharadwaj-chukkala/Stereo-Vision-to-estimate-depth-in-an-image/blob/master/results/disparity_image_heat1.png) | ![disp2](https://github.com/bharadwaj-chukkala/Stereo-Vision-to-estimate-depth-in-an-image/blob/master/results/disparity_image_heat2.png)     | ![disp3](https://github.com/bharadwaj-chukkala/Stereo-Vision-to-estimate-depth-in-an-image/blob/master/results/disparity_image_heat3.png)     |
| Depth Estimation                   | ![depth1](https://github.com/bharadwaj-chukkala/Stereo-Vision-to-estimate-depth-in-an-image/blob/master/results/depth_image_heat1.png)            | ![depth2](https://github.com/bharadwaj-chukkala/Stereo-Vision-to-estimate-depth-in-an-image/blob/master/results/depth_image_heat2.png)        | ![depth3](https://github.com/bharadwaj-chukkala/Stereo-Vision-to-estimate-depth-in-an-image/blob/master/results/depth_image_heat3.png)        |

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References
- [3D DISTANCE MEASUREMENT ACCURACY ON LOW-COST STEREOCAMERA, M.F. Abu Hassan, A. Hussain, M.H. Md Saad, K. Win, Sci.Int.(Lahore),29(3),599-605,2017 ISSN: 1013-5316; CODEN: SINTE 8](https://www.researchgate.net/publication/318452089_3D_DISTANCE_MEASUREMENT_ACCURACY_ON_LOW-COST_STEREO_CAMERA/figures?lo=1)

## Contact

**Bharadwaj Chukkala**<br>
UID: 118341705<br>
Bharadwaj Chukkala is currently a Master's student in Robotics at the University of Maryland, College Park, MD (Batch of 2023). His interests include Machine Learning, Perception and Path Planning.<br>
[![Contact](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](bchukkal@umd.edu)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/bharadwaj-chukkala/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/bharadwaj-chukkala)

