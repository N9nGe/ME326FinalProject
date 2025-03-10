#!/usr/bin/env python3

"""
script for aligning rgb and depth images. Tested on the Realsense L415 camera. Might need to adjust the intrinsics for other cameras. 

NOTE: This code only serves as a reference and might need to be adjusted to work in your case!

copyright Armlab@Stanford
"""

import numpy as np
import cv2
from typing import Tuple


def convert_intrinsics(img, K_old, K_new, new_size=(1280, 720)):
    """
    Convert a set of images to a different set of camera intrinsics.
    Parameters:
    - images: List of input images.
    - K_old: Matrix of the old camera intrinsics.
    - K_new: Matrix of the new camera intrinsics.
    - new_size: Tuple (width, height) defining the size of the output images.
    Returns:
    - List of images converted to the new camera intrinsics.
    """
    width, height = new_size
    
    # Compute the inverse of the new intrinsics matrix for remapping
    K_new_inv = np.linalg.inv(K_new)
    
    # Construct a grid of points representing the new image coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    homogenous_coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x).ravel()], axis=-1).T
    
    # Convert to the old image coordinates
    old_coords = K_old @ K_new_inv @ homogenous_coords
    old_coords /= old_coords[2, :]  # Normalize to make homogeneous
    print("old coords: ", old_coords)
    # Reshape for remapping
    map_x = old_coords[0, :].reshape(height, width).astype(np.float32)
    map_y = old_coords[1, :].reshape(height, width).astype(np.float32)
    
    # Remap the image to the new intrinsics
    converted_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return converted_img


def warp_image(image, K, R, t):
    """
    Warp an image from the perspective of camera 1 to camera 2.

    :param image: Input image from camera 1
    :param K: Intrinsic matrix of both cameras
    :param R: Rotation matrix from camera 1 to camera 2
    :param t: Translation vector from camera 1 to camera 2
    :return: Warped image as seen from camera 2
    """
    # Compute the homography matrix
    H = compute_homography(K, R, t)

    # Warp the image using the homography
    height, width = image.shape[:2]
    warped_image = cv2.warpPerspective(image, H, (width, height))

    return warped_image


def compute_homography(K, R, t):
    """
    Compute the homography matrix given intrinsic matrix K, rotation matrix R, and translation vector t.
    """
    K_inv = np.linalg.inv(K)
    H = np.dot(K, np.dot(R - np.dot(t.reshape(-1, 1), K_inv[-1, :].reshape(1, -1)), K_inv))
    return H


def align_depth(depth: np.ndarray, depth_K: Tuple[float,float,float,float], rgb: np.ndarray, rgb_K: Tuple[float,float,float,float], cam2cam_transform: np.ndarray) -> np.ndarray:
    """
    align depth image to the rgb image.

    :param depth: depth image
    :param depth_K: intrinsics of the depth camera
    :param rgb: rgb image
    :param rgb_K: intrinsics of the rgb camera
    :param cam2cam_transform: transformation matrix from depth to rgb camera
    :return: aligned depth image
    """
    
    old_fx, old_fy, old_cx, old_cy = depth_K
    new_fx, new_fy, new_cx, new_cy = rgb_K
    # Constructing the old and new intrinsics matrices
    K_old = np.array([[old_fx, 0, old_cx], [0, old_fy, old_cy], [0, 0, 1]])
    K_new = np.array([[new_fx, 0, new_cx], [0, new_fy, new_cy], [0, 0, 1]])
    print("K_old: ", K_old)
    print("K_new: ", K_new)
    print("fuck u sam")

    # conver the instrinsics of the depth camera to the intrinsics of the rgb camera.
    depth = convert_intrinsics(depth, K_old, K_new, new_size=(rgb.shape[1], rgb.shape[0]))
    print("depth: ", depth)
    # warp the depth image to the rgb image with the transformation matrix from camera to camera.
    depth = warp_image(depth, K_new, cam2cam_transform[:3, :3], cam2cam_transform[:3, 3])        
    return depth


"""
sample usage

depth = cv2.imread("depth.png", cv2.IMREAD_UNCHANGED)
rgb = cv2.imread("rgb.png")
depth_K = (360.01, 360.01, 243.87, 137.92)
rgb_K = (1297.67, 1298.63, 620.91, 238.28)
cam2cam_transform = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

aligned_depth = align_depth(depth, depth_K, rgb, rgb_K, cam2cam_transform)
cv2.imwrite("aligned_depth.png", aligned_depth)

"""                                                           