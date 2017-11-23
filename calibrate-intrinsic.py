#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function

import os, math, cv2, sys
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
subpix_stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

fisheye_calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW

def reprojectionErrors(img_points, pattern_corners, rvec, tvec, camera_matrix, dist_coeffs, fisheye=False):
  """
  Compute the distances between
    1. the reprojection of chessboard corners projected through a given calibration and
    2. the chessboard corners detected on the image
  """

  if fisheye:
    reprojected_points, _ = cv2.fisheye.projectPoints(pattern_corners.reshape(1, -1, 3), rvec, tvec, camera_matrix, dist_coeffs)
  else:
    reprojected_points, _ = cv2.projectPoints(pattern_corners, rvec, tvec, camera_matrix, dist_coeffs)

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

def saveImage(image, filename, prefix, output_dir):
  new_filename = os.path.join(output_dir, prefix + '_' + os.path.basename( filename ))
  print("saving:", new_filename)
  cv2.imwrite(new_filename, image)

def detectCorners(gray_image, corners_size, subpix_window_size):

  found, corners = cv2.findChessboardCorners(gray_image, corners_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

  if not found:
    return (False, None)

  cv2.cornerSubPix(gray_image, corners, subpix_window_size, (-1, -1), subpix_stop_criteria)

  return (True, np.asarray(corners, dtype='float32').reshape(1, -1, 2))

def calibratePatterns(img_points, pattern_corners, img_size, n_corners, fisheye=False):

  # TODO add optional initial guess

  # number of detected patterns
  n_patterns = len( img_points )

  # the pattern corners are the same for each image. create one for each
  # detected pattern and reshape everything to the expected shape.
  obj_points = np.asarray([ pattern_corners ] * n_patterns, dtype=np.float32).reshape(n_patterns, 1, -1, 3)

  camera_matrix = np.eye( 3 )
  dist_coeffs = np.zeros( 4 )

  rvecs = np.zeros((n_patterns, 1, 1, 3), dtype=np.float64)
  tvecs = np.zeros((n_patterns, 1, 1, 3), dtype=np.float64)

  if fisheye:

    return cv2.fisheye.calibrate(obj_points, img_points, img_size, camera_matrix, dist_coeffs, rvecs, tvecs, fisheye_calibration_flags, calibration_stop_criteria)

  else:

    return cv2.calibrateCamera(obj_points, img_points, img_size, camera_matrix, dist_coeffs, rvecs, tvecs)

def showErrors(pattern_corners, img_points, camera_matrix, dist_coeffs, rvecs, tvecs, filenames, fisheye=False):

  fig = plt.figure()
  ax = fig.add_subplot(111)

  limit_circle = plt.Circle((0, 0), 1.0, color='r', linestyle='--', fill=False, alpha=0.5)
  ax.add_artist( limit_circle )

  print()
  print("rmse values for individual images:")

  for i in range(len(img_points)):

    errors = reprojectionErrors(img_points[i], pattern_corners, rvecs[i], tvecs[i], camera_matrix, dist_coeffs, fisheye)

    img_label, _ = os.path.splitext(os.path.basename(filenames[i]))
    ax.scatter(errors[:,0], errors[:,1], marker='+', label=img_label)

    print(img_label + ":", computeRMSE(errors))

  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles, labels)

  ax.set_xlabel('x (px)')
  ax.set_ylabel('y (px)')

  ax.set_aspect('equal')

  #~ xmin, xmax = ax.get_xlim()
  #~ ax.set_xlim( min(xmin, -1.1), max(1.1, xmax) )

  #~ ymin, ymax = ax.get_ylim()
  #~ ax.set_ylim( min(ymin, -1.1), max(1.1, ymax) )

  fig.suptitle("reprojection errors")

def undistortImages(filenames, camera_matrix, dist_coeffs, output_dir, fisheye=False):

  for filename in filenames:

    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    h, w = image.shape[:2]

    if fisheye:

      assert( False )

    else:

      #~ new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
      #~ image_undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, newCameraMatrix=new_camera_matrix)
      image_undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
      
    ################
    ## save image ##
    ################

    saveImage(image_undistorted, filename, "undistorted", output_dir)

def main():

  ####################
  ## load arguments ##
  ####################

  import argparse

  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("filenames", nargs='+', type=str, help="OpenCV VideoCapture input format")
  parser.add_argument('-o', '--output_dir', type=str, default="output", help="output directory for debug images.")
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

  # prepare debug directory
  if args.debug and not os.path.isdir( args.output_dir ):
    os.mkdir( args.output_dir )

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

  ###############################
  ## detect chessboard corners ##
  ###############################

  # 2d points in image plane.
  img_points = []

  # image size will be read from images and checked to be consistent.
  img_size = None

  for filename in args.filenames:

    image = cv2.imread(filename, cv2.IMREAD_COLOR)

    if image is None:
      print("Error: image", filename, "could not be read")
      return 1

    if img_size is None:
      img_size = image.shape[:2][::-1]
      #print("image size:", img_size)

    assert img_size == image.shape[:2][::-1], "All images must share the same size."

    ####################
    ## detect corners ##
    ####################

    # findChessboardCorners doesn't need to be grayscale, but cornerSubPix does
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    found, corners = detectCorners(gray_image, corners_size, subpix_window_size)

    if not found:
      print("Warning: no checkerboard found on image")
      continue

    img_points.append( corners )

    ###########################
    ## show detected corners ##
    ###########################

    if args.debug:
      image = cv2.drawChessboardCorners(image, corners_size, corners, found)
      saveImage(image, filename, "corners", args.output_dir)

  ######################
  ## Calibrate camera ##
  ######################

  rmse, camera_matrix, dist_coeffs, rvecs, tvecs = calibratePatterns(img_points, pattern_corners, img_size, n_corners, args.fisheye)

  ##############################
  ## show reprojection errors ##
  ##############################

  showErrors(pattern_corners, img_points, camera_matrix, dist_coeffs, rvecs, tvecs, args.filenames, args.fisheye)

  # same metric as used for single image rmse
  print()
  print("calibration rmse (opencv):", rmse)

  ##################
  ## Save results ##
  ##################

  print()
  print("camera_matrix:", toYamlArray(camera_matrix))
  print("dist_coeff:", toYamlArray(dist_coeffs))
  print()

  #############################
  ## show rectified patterns ##
  #############################

  if args.debug:
    undistortImages(args.filenames, camera_matrix, dist_coeffs, args.output_dir, args.fisheye)

  plt.show()

  return 0

if __name__ == '__main__':
  import sys
  sys.exit( main() )
