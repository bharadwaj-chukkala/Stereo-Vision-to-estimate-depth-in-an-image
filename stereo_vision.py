import numpy as np
import cv2
import argparse
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

'''
Function that creates SIFT features (point correspondences) to an array
Returns :  An array of corresponding pairs of points
'''

def SIFT_features_as_array(sift_matches, kp1, kp2):
    matching_pairs = []
    for i, m1 in enumerate(sift_matches):
        pts_1 = kp1[m1.queryIdx].pt
        pts_2 = kp2[m1.trainIdx].pt
        matching_pairs.append([pts_1[0], pts_1[1], pts_2[0], pts_2[1]])
        array_pair = np.array(matching_pairs).reshape(-1, 4)
    matching_pairs = array_pair
    return matching_pairs

'''
Function to func_normalize_val the value of xy
Returns : normalized x value
'''
def func_normalize_val(xy):

    xy_new = np.mean(xy, axis=0)
    x_new ,y_dash = xy_new[0], xy_new[1]

    x_hat = xy[:,0] - x_new
    y_hat = xy[:,1] - y_dash
    
    dist = np.mean(np.sqrt(x_hat**2 + y_hat**2))
    s = (2/dist)
    
    scaling = np.diag([s,s,1])
    T_trans = np.array([[1,0,-x_new],[0,1,-y_dash],[0,0,1]])
    T = scaling.dot(T_trans)

    x_ = np.column_stack((xy, np.ones(len(xy))))
    x_norm = (T.dot(x_.T)).T

    return  x_norm, T


'''
Fucntion to retrieve x value from a line equation
Returns x from y=mx+c
'''
def retrieve_X(line, y):
    a= line[0]
    b = line[1]
    c = line[2]
    x = -(b*y + c)/a
    return x

'''
Function to compute epi polar geometry
Returns epipolar lines
'''
def Compute_epi_lines(pts_set1, pts_set2, F, image0, image1, filename, rectified = False):
    
    lines1, lines2 = [], []
    epipolar_img1 = image0.copy()
    epipolar_img2 = image1.copy()

    for i in range(pts_set1.shape[0]):
        x1 = np.array([pts_set1[i,0], pts_set1[i,1], 1]).reshape(3,1)
        x2 = np.array([pts_set2[i,0], pts_set2[i,1], 1]).reshape(3,1)

        line2 = np.dot(F, x1)
        lines2.append(line2)

        line1 = np.dot(F.T, x2)
        lines1.append(line1)
    
        if not rectified:
            y2_min = 0
            y2_max = image1.shape[0]
            x2_min = retrieve_X(line2, y2_min)
            x2_max = retrieve_X(line2, y2_max)

            y1_min = 0
            y1_max = image0.shape[0]
            x1_min = retrieve_X(line1, y1_min)
            x1_max = retrieve_X(line1, y1_max)
        else:
            x2_min = 0
            x2_max = image1.shape[1] - 1
            y2_min = -line2[2]/line2[1]
            y2_max = -line2[2]/line2[1]

            x1_min = 0
            x1_max = image0.shape[1] -1
            y1_min = -line1[2]/line1[1]
            y1_max = -line1[2]/line1[1]


        cv2.circle(epipolar_img2, (int(pts_set2[i,0]),int(pts_set2[i,1])), 10, (0,0,255), -1)
        cv2.circle(epipolar_img1, (int(pts_set1[i,0]),int(pts_set1[i,1])), 10, (0,0,255), -1)
        
        epipolar_img2 = cv2.line(epipolar_img2, (int(x2_min), int(y2_min)), (int(x2_max), int(y2_max)), (0, 255, 0), 2)
        epipolar_img1 = cv2.line(epipolar_img1, (int(x1_min), int(y1_min)), (int(x1_max), int(y1_max)), (0, 255, 0), 2)
        
        images=[epipolar_img1, epipolar_img2]
        sizes = []
        for image in images:
            x, y, ch = image.shape
            sizes.append([x, y, ch])
        
        sizes = np.array(sizes)
        x_target, y_target, _ = np.max(sizes, axis = 0)
        
        images_resized = []

        for i, image in enumerate(images):
            image_resized = np.zeros((x_target, y_target, sizes[i, 2]), np.uint8)
            image_resized[0:sizes[i, 0], 0:sizes[i, 1], 0:sizes[i, 2]] = image
            images_resized.append(image_resized)
            

    image_1, image_2 = images_resized
    
    img_group = np.concatenate((image_1, image_2), axis = 1)
    img_group = cv2.resize(img_group, (1920, 660))

    plt.imshow(img_group)
    plt.savefig(filename)
    
    return lines1, lines2

'''
Function to compute the Fundamental Matrix for the Dataset
Returns the estimated Fundamental Matrix
'''

def Compute_Fundamental_Matrix(feature_matches):
    
    normalised = True
    thresh_val = 7

    x1 = feature_matches[:,0:2]
    x2 = feature_matches[:,2:4]

    if x1.shape[0] > thresh_val:
        if normalised == True:
            x1_norm, T1 = func_normalize_val(x1)
            x2_norm, T2 = func_normalize_val(x2)
        else:
            x1_norm,x2_norm = x1,x2
            
        A = np.zeros((len(x1_norm),9))
        for i in range(0, len(x1_norm)):
            x_1,y_1 = x1_norm[i][0], x1_norm[i][1]
            x_2,y_2 = x2_norm[i][0], x2_norm[i][1]
            A[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2*x_1, y_2*y_1, y_2, x_1, y_1, 1])

        U, S, VT = np.linalg.svd(A, full_matrices=True)
        F = VT.T[:, -1]
        F = F.reshape(3,3)

        u, s, vt = np.linalg.svd(F)
        s = np.diag(s)
        s[2,2] = 0
        F = np.dot(u, np.dot(s, vt))

        if normalised:
            F = np.dot(T2.T, np.dot(F, T1))
        return F

    else:
        return None

'''
Function to calculate the absolute error between the corresponding features
Returns absolute difference
'''
def epipolar_constraint(feature, F): 
    x1,x2 = feature[0:2], feature[2:4]
    x1tmp=np.array([x1[0], x1[1], 1]).T
    x2tmp=np.array([x2[0], x2[1], 1])

    error = np.dot(x1tmp, np.dot(F, x2tmp))
    
    return np.abs(error)

'''
Function to randomly select points and get a best fit of features
Returns filtered features
'''
def RANSAC(features):
    ##-- parameters --##
    iterations = 1000
    threshold_val = 0.02
    inliers_thresh = 0
    best_fit_pts = []
    fundamental_mat = 0

    for i in range(0, iterations):
        indices = []
        #select 8 points randomly and check for best fit
        n_rows = features.shape[0]
        random_indices = np.random.choice(n_rows, size=8)
        best_correspondences = features[random_indices, :] 
        f_8 = Compute_Fundamental_Matrix(best_correspondences)
        
        #---- thresholding error indices ---#
        for j in range(n_rows):
            feature = features[j]
            error = epipolar_constraint(feature, f_8)
            if error < threshold_val:
                indices.append(j)
        
        #--- choosing best fundamental matrix ---#
        if len(indices) > inliers_thresh:
            inliers_thresh = len(indices)
            best_fit_pts = indices
            fundamental_mat = f_8

    required_features = features[best_fit_pts, :]
    return fundamental_mat, required_features

'''
Function to compute Essential Matrix
Returns Essentialmatrix with rank 2
'''
def compute_Essential_matrix(K1, K2, F):
    E = K2.T.dot(F).dot(K1)
    U,S,V = np.linalg.svd(E)
    #rank constraint
    S = [1,1,0]
    new_E = np.dot(U,np.dot(np.diag(S),V))
    return new_E


'''
Function to restore the camera pose configuration
Return best pose configuration
'''
def Restore_cam_pose(E):
    '''
    Conceptual basis from : https://cmsc733.github.io/2022/proj/p3/#estfundmatrix
    '''
    U, S, V_T = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    Rot = []
    trans = []
    
    ###--- 4 Rotational configurations ---###
    Rot.append(np.dot(U, np.dot(W, V_T)))
    Rot.append(np.dot(U, np.dot(W, V_T)))
    Rot.append(np.dot(U, np.dot(W.T, V_T)))
    Rot.append(np.dot(U, np.dot(W.T, V_T)))
    
    ###-- 4 translational configurations ---###
    trans.append(U[:, 2])
    trans.append(-U[:, 2])
    trans.append(U[:, 2])
    trans.append(-U[:, 2])

    ###--- Retrieving the best cam pose config ---###
    for i in range(4):
        if (np.linalg.det(Rot[i]) < 0):
            Rot[i] = -Rot[i]
            trans[i] = -trans[i]

    return Rot, trans


'''
Function to check the cheirality condition to Recover the pose
Returns the pixels infront of the camera
'''
def check_cheirality(pts_3D,trans,Rot):
    best_pts = 0
    for P in pts_3D:
        P = P.reshape(-1,1)
        if Rot.dot(P - trans) > 0 and P[2]>0:
            best_pts+=1
    return best_pts


'''
Main fucntion to call all existing  fucntions
Returns:Fundamental Matrix,
        Essential Matrix,
        Homographies of images,
        Plots:
            Epipolar lines
            Disparity maps
            Depth maps
'''
def main():
    
###################################################################
##---------------------READ DATA---------------------------------##
###################################################################  
    
    print("Dataset curule = 1 \nDataset octagon = 2 \nDataset pendulum = 3")
    dataset_number = input("Enter the dataset number: ")
    dataset_number = int(dataset_number)
    
    
    if dataset_number == 1:
        K1 = np.array([[1758.23, 0, 977.42],[ 0, 1758.23, 552.15],[ 0, 0, 1]])
        K2 = K1
        params = [0,88.39,1920,1080,220,55,195]
        baseline = params[1]
        f = K1[0,0]
        depth_thresh = 100000
        
        img1 = cv2.imread('data/curule/im0.png')
        img2 = cv2.imread('data/curule/im1.png')
        window = 5

    elif dataset_number == 2:
        K1 = np.array([[1742.11, 0, 804.90],[ 0, 1742.11, 541.22],[ 0, 0, 1]])
        K2 = K1
        params = [0,221.76,1920,1080,100,29,61]
        baseline = params[1]
        f = K1[0,0]
        depth_thresh = 1000000
        img1 = cv2.imread('data/octagon/im0.png')
        img2 = cv2.imread('data/octagon/im1.png')
        window = 7

    elif dataset_number == 3:
        K1 = np.array([[1729.05, 0, -364.24],[ 0, 1729.05, 552.22],[ 0, 0, 1]])
        K2 = K1
        params = [0,537.75,1920,1080,100,25,150]
        baseline = params[1]
        f = K1[0,0]
        depth_thresh = 100000
        img1 = cv2.imread('data/pendulum/im0.png')
        img2 = cv2.imread('data/pendulum/im0.png')
        window = 3

    else:
        print("invalid datset number")
        
    print('Starting Stereo Vision')
    

###################################################################
##---------------------CALIBRATION ------------------------------##
###################################################################   
    
    ###--- SIFT--- ###
    sift = cv2.SIFT_create()
    image0 = img1
    image1 = img2

    image0_gray = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY) 
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    image0_rgb = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB) 
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    
    print("Finding matches\n")
    kp1, des1 = sift.detectAndCompute(image0_gray, None)
    kp2, des2 = sift.detectAndCompute(image1_gray, None)
    
    ###-- Brute Force Matching --###
    bf = cv2.BFMatcher()
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x :x.distance)
    chosen_matches = matches[0:100]
    
    print("Matches Found and sorted\n")
    
    matched_image = cv2.drawMatches(image0_rgb,kp1,image1_rgb,kp2,matches[:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(matched_image)
    plt.savefig('results/curule_matched_image.png')
    
    matched_pairs = SIFT_features_as_array(chosen_matches, kp1, kp2)
    print("Estimating Fundamental and Essential matrix \n")
    
    ###-- Computing Fundamental Matrix ---###
    F_best, best_correspondoing_points = RANSAC(matched_pairs)
    

    ###---Computing Essential Matrix ---###
    E = compute_Essential_matrix(K1, K2, F_best)
    
    print("Dataset ", dataset_number  ,":\n")
    print('The Fundamental Matrix: \n',F_best,'\n')
    print('The Essential Matrix: \n', E,'\n')
    
    R2_, C2_ = Restore_cam_pose(E)
    
    Pts_3D = []
    R1  = np.identity(3)
    C1  = np.zeros((3, 1))
    I = np.identity(3)

    ####---- Triangulation----####
    for i in range(len(R2_)):
        R2 =  R2_[i]
        C2 =   C2_[i].reshape(3,1)
        Proj_mat_im0 = np.dot(K1, np.dot(R1, np.hstack((I, -C1.reshape(3,1)))))
        Proj_mat_im1 = np.dot(K2, np.dot(R2, np.hstack((I, -C2.reshape(3,1)))))

        for x_left_img,x_right_img in zip(best_correspondoing_points[:,0:2], best_correspondoing_points[:,2:4]):

            pts_3d = cv2.triangulatePoints(Proj_mat_im0, Proj_mat_im1, np.float32(x_left_img), np.float32(x_right_img))
            pts_3d = np.array(pts_3d)
            pts_3d = pts_3d[0:3,0]
            Pts_3D.append(pts_3d) 

    best_indices = 0
    max_Positive = 0

    ### --Camera Pose Restoration--###
    for i in range(len(R2_)):
        R_, C_ = R2_[i],  C2_[i].reshape(-1,1)
        R_3 = R_[2].reshape(1,-1)
        num_Positive = check_cheirality(Pts_3D,C_,R_3)

        if num_Positive > max_Positive:
            best_indices = i
            max_Positive = num_Positive

    Rotation_coniguration, Translation_coniguration, P3D = R2_[best_indices], C2_[best_indices], Pts_3D[best_indices]

    print(" Camera Pose: (Rotation) \n",Rotation_coniguration,'\n')
    print(" Camera Pose: (Translation) \n", Translation_coniguration, '\n')
    
    
###################################################################
##----------------- RECTIFICATION--------------------------------##
###################################################################
    
    pts_set1,pts_set2= best_correspondoing_points[:,0:2], best_correspondoing_points[:,2:4]
    
    lines1, lines2 = Compute_epi_lines(pts_set1, pts_set2, F_best, image0, image1, "results/epi_polar_lines_" + str(dataset_number)+ ".png", False)
    
    h1, w1 = image0.shape[:2]
    h2, w2 = image1.shape[:2]
    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts_set1), np.float32(pts_set2), F_best, imgSize=(w1, h1))
    print("Estimated H1 and H2 as \n \n Homography Matrix 1: \n", H1,'\n \n Homography Matrix 2:\n ', H2)
    
    img1_rectified = cv2.warpPerspective(image0, H1, (w1, h1))
    img2_rectified = cv2.warpPerspective(image1, H2, (w2, h2))
       
    
    pts_set1_rectified = cv2.perspectiveTransform(pts_set1.reshape(-1, 1, 2), H1).reshape(-1,2)
    pts_set2_rectified = cv2.perspectiveTransform(pts_set2.reshape(-1, 1, 2), H2).reshape(-1,2)
    
    H2_T_inv =  np.linalg.inv(H2.T)
    H1_inv = np.linalg.inv(H1)
    F_rectified = np.dot(H2_T_inv, np.dot(F_best, H1_inv))
    
    lines1_rectified, lines2_rectified = Compute_epi_lines(pts_set1_rectified, pts_set2_rectified, F_rectified, img1_rectified, img2_rectified, "results/rectified_epi_polar_lines_" + str(dataset_number)+ ".png",  True)
    
    img1_rectified_reshaped = cv2.resize(img1_rectified, (int(img1_rectified.shape[1] / 4), int(img1_rectified.shape[0] / 4)))
    img2_rectified_reshaped = cv2.resize(img2_rectified, (int(img2_rectified.shape[1] / 4), int(img2_rectified.shape[0] / 4)))
    
    img1_rectified_reshaped = cv2.cvtColor(img1_rectified_reshaped, cv2.COLOR_BGR2GRAY)
    img2_rectified_reshaped = cv2.cvtColor(img2_rectified_reshaped, cv2.COLOR_BGR2GRAY)
    
    
###################################################################
##----------------- CORRESPONDENCE ------------------------------##
###################################################################


    #Taking the rectified images as input to calculate the disparity
    img_rect_left, img_rect_right = img1_rectified_reshaped, img2_rectified_reshaped
    
    img_rect_left = img_rect_left.astype(int)
    img_rect_right = img_rect_right.astype(int)

    height, width = img_rect_left.shape
    disparity_map = np.zeros((height, width))

    x_new = width - (2 * window)
    
    ### ---- Block Matching --- ###
    for y in tqdm(range(window, height-window)):
        
        block_img_rect_left = []
        block_img_rect_right = []
        for x in range(window, width-window):
            block_left = img_rect_left[y:y + window, x:x + window]
            block_img_rect_left.append(block_left.flatten())

            block_right = img_rect_right[y:y + window, x:x + window]
            block_img_rect_right.append(block_right.flatten())

        block_img_rect_left = np.array(block_img_rect_left)
        block_img_rect_right = np.array(block_img_rect_right)
        
        block_img_rect_left = np.repeat(block_img_rect_left[:, :, np.newaxis], x_new, axis=2)
        block_img_rect_right = np.repeat(block_img_rect_right[:, :, np.newaxis], x_new, axis=2)
        

        block_img_rect_right = block_img_rect_right.T
        
        ###----Sum of Absolute Differences (SAD)----###
        absolute_difference = np.abs(block_img_rect_left - block_img_rect_right)
        SAD = np.sum(absolute_difference, axis = 1)
        index = np.argmin(SAD, axis = 0)
        disparity = np.abs(index - np.linspace(0, x_new, x_new, dtype=int)).reshape(1, x_new)
        disparity_map[y, 0:x_new] = disparity 


    ###-- Plotting Disparity Maps ---#
    disparity_map_int = np.uint8(disparity_map * 255 / np.max(disparity_map))
    plt.imshow(disparity_map_int, cmap='hot', interpolation='nearest')
    plt.savefig('results/disparity_image_heat' +str(dataset_number)+ ".png")
    plt.imshow(disparity_map_int, cmap='gray', interpolation='nearest')
    plt.savefig('results/disparity_image_gray' +str(dataset_number)+ ".png")
    print('Disparity Maps Plotted')

###################################################################
##-------------------------- DEPTH ------------------------------##
###################################################################
    
    ###--- Thresholding depth values---###
    if(dataset_number == 1):
        depth = (baseline * f) / (disparity_map + 1e-10)
        depth[depth > depth_thresh] = depth_thresh
    elif(dataset_number == 2):
        depth = (baseline * f) / (disparity_map + 1e-10)
        depth[depth > depth_thresh] = depth_thresh
    elif(dataset_number == 3):
        depth = (baseline * f) / (disparity_map + 1e-10)
        depth[depth > depth_thresh] = depth_thresh
    else:
        print("Invalid dataset number")
    
    ##--- Plotting Depth Maps ---##
    depth_map = np.uint8(depth * 255 / np.max(depth))
    plt.imshow(depth_map, cmap='hot', interpolation='nearest')
    plt.savefig('results/depth_image_heat' +str(dataset_number)+ ".png")
    plt.imshow(depth_map, cmap='gray', interpolation='nearest')
    plt.savefig('results/depth_image_gray' +str(dataset_number)+ ".png")
    plt.show()
    print('Depth Maps Plotted')
    print('Stereo Vision Completed')


    
if __name__ == '__main__':
    main()