#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import math
import cv2, sys
import numpy as np

import matplotlib.pyplot as plt

"""
TermCriteria(int type, int maxCount, double epsilon)
 - type: The type of termination criteria, one of { COUNT | MAX_ITER | EPS }.
    * COUNT: the maximum number of iterations or elements to compute
    * MAX_ITER: ditto
    * EPS: the desired accuracy or change in parameters at which the iterative algorithm stops 
 - maxCount: The maximum number of iterations or elements to compute.
 - epsilon: The desired accuracy or change in parameters at which the iterative algorithm stops.
"""
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

def reprojectionErrors(img_points, obj_points, rvec, tvec, camera_matrix, dist_coeffs):
  """
  Compute the distances between
    1. the reprojection of chessboard corners projected through a given calibration and
    2. the chessboard corners detected on the image
  """
  reprojected_points, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
  reprojected_points = reprojected_points.reshape(1, -1, 2)
  return (img_points - reprojected_points).reshape(-1, 2)

def computeRMSE(errors):
  """
  @brief compute root mean square error for a collection of 2D error (distance) tuples.
  @return square root of the sum of squared errors
      \sqrt{\frac{\sum_{i}^{n}e\left(\mathbf{x}_{i},\hat{\mathbf{x}}_{i}\right)^{2}}{n}}
    where error is defined as the euclidean distance
      e\left(\mathbf{x},\hat{\mathbf{x}}\right)=\sqrt{\left(u-\hat{u}\right)^{2}+\left(v-\hat{v}\right)^{2}}
  """
  errors_flat = errors.ravel()
  return math.sqrt( errors_flat.dot( errors_flat ) / len( errors ) )

def toYamlArray(m):
  return "[ " + ", ".join( map(str, m.ravel()) ) + " ]"

def main():

  ####################
  ## load arguments ##
  ####################

  import argparse

  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("capture", type=str, help="OpenCV VideoCapture input format")
  parser.add_argument('-r', '--rows', type=int, default=6, help="number of calibration chessboard rows")
  parser.add_argument('-c', '--cols', type=int, default=9, help="number of calibration chessboard columns")
  parser.add_argument('-w', '--window_size', type=int, default=5, help="half size of the subpixel search window (in px).")
  parser.add_argument('-s', '--square_size', type=float, default=0.025, help="dimentions of a single square of the chessboard")
  parser.add_argument('-f', '--fisheye', action='store_true', help="use fisheye distortion model")
  parser.add_argument('-d', '--debug', action='store_true', help="show detected corners")
  args = parser.parse_args()

  ##########################
  ## setup some variables ##
  ##########################

  if args.fisheye:
    assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'

    # number of internal rows and columns of corners
  board_rows = args.rows
  board_cols = args.cols
  n_corners = board_rows * board_cols
  corners_size = (board_rows, board_cols)

  xx, yy = np.meshgrid(range(board_rows), range(board_cols))
  pattern_corners = np.array(zip(xx.ravel(), yy.ravel(), np.zeros(n_corners)), dtype=np.float64) * args.square_size

  """
  Half of the side length of the search window (in px).
  For example, if winSize=Size(5,5), then a 5∗2+1×5∗2+1=11×11 search window is used.
  """
  subpix_window_size = (args.window_size, args.window_size)

  #########################
  ## open capture device ##
  #########################

  capture = cv2.VideoCapture( args.capture )

  if not capture.isOpened():
    print("Error opening video capture")
    return 1

  obj_points = []
  img_points = []

  # image size will be read from images and checked to be consistent.
  img_size = None

  while capture.isOpened():

    ret, image = capture.read()

    # no image means end of capture
    if not ret:
      break

    if img_size is None:
      img_size = image.shape[:2][::-1]
      #print("image size:", img_size)

    assert img_size == image.shape[:2][::-1], "All images must share the same size."

    ####################
    ## detect corners ##
    ####################

    # findChessboardCorners doesn't need to be grayscale, but cornerSubPix does
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(image, corners_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if not found:
      print("Warning: no checkerboard found on image")
      continue

    # nota: con esto anda peor, tal vez modificando el criteria?
    # https://docs.opencv.org/3.3.1/dd/d1a/group__imgproc__feature.html#ga354e0d7c86d0d9da75de9b9701a9a87e
    cv2.cornerSubPix(image, corners, subpix_window_size, (-1, -1), subpix_criteria)

    obj_points.append( pattern_corners )
    img_points.append( corners )

    ###########################
    ## show detected corners ##
    ###########################

    if args.debug:

      cv2.drawChessboardCorners(image, corners_size, corners, found)

      cv2.imshow('distorted', image)
      cv2.waitKey( 0 )

  capture.release()
  cv2.destroyAllWindows()

  ######################
  ## Calibrate camera ##
  ######################

  # number of detected patterns
  n_patterns = len(img_points)
  #print("found", n_patterns, "patterns")

  # initial guess
  #~ camera_matrix = np.array([
    #~ [872.34347227, 0., 951.22313648],
    #~ [0. , 881.1423476, 545.82512402],
    #~ [0., 0., 1.]
  #~ ])

  camera_matrix = np.eye( 3 )
  dist_coeffs = np.zeros( 4 )

  if args.fisheye:

    obj_points = np.asarray([obj_points], dtype='float64').reshape(-1, 1, n_corners, 3)
    img_points = np.asarray([img_points], dtype='float64').reshape(-1, 1, n_corners, 2)

    rmse, camera_matrix, dist_coeffs, _, _ = cv2.fisheye.calibrate(obj_points, img_points, img_size, camera_matrix, dist_coeffs)

  else:

    obj_points = np.asarray([obj_points], dtype='float32').reshape(-1, 1, n_corners, 3)
    img_points = np.asarray([img_points], dtype='float32').reshape(-1, 1, n_corners, 2)

    rvecs = np.zeros((n_patterns, 1, 1, 3), dtype=np.float32)
    tvecs = np.zeros((n_patterns, 1, 1, 3), dtype=np.float32)

    # doc: https://docs.opencv.org/3.1.0/d9/d0c/group__calib3d.html#ga687a1ab946686f0d85ae0363b5af1d7b
    rmse, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, camera_matrix, dist_coeffs, rvecs, tvecs)

  ##############################
  ## show reprojection errors ##
  ##############################

  fig = plt.figure()
  ax = fig.add_subplot(111)

  print("rmse values for individual images:")

  for i in range(n_patterns):

    errors = reprojectionErrors(img_points[i], pattern_corners, rvecs[i], tvecs[i], camera_matrix, dist_coeffs)

    ax.scatter(errors[:,0], errors[:,1], marker='+', label=str(i))

    print(i, "\t:", computeRMSE(errors))

  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles, labels)

  ax.set_xlabel('x (px)')
  ax.set_ylabel('y (px)')

  fig.suptitle("reprojection errors")

  # same metric as used for single image rmse
  print()
  print("calibration rmse (opencv):", rmse)

  ##################
  ## Save results ##
  ##################

  print()
  print("camera_matrix:", toYamlArray(camera_matrix))
  print("dist_coeff:", toYamlArray(dist_coeffs))

  #~ import yaml
  #~ with open("calib.yml", "w") as f:
    #~ yaml.dump({"camera_matrix": camera_matrix, "dist_coeff": dist_coeffs}, f)

  #############################
  ## show rectified patterns ##
  #############################

  # Can't read twice from the same capture :(
  """
  capture = cv2.VideoCapture( args.capture )

  if not capture.isOpened():
    print("Error opening video capture")
    return 1

  while capture.isOpened():

    ret, image = capture.read()

    if not ret:
      break

      h,  w = image.shape[:2]

      if args.fisheye:

        assert( false )

      else:

        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        image_undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, newCameraMatrix=new_camera_matrix)

        cv2.imshow('undistorted', image_undistorted)
        cv2.waitKey( 0 )

  cv2.destroyAllWindows()
  """

  plt.show()

  return 0

if __name__ == '__main__':
  import sys
  sys.exit( main() )