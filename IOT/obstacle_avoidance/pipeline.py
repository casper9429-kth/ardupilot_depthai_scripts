#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time

import blobconverter

def create_pipeline():
    # Start defining a pipeline
    pipeline = dai.Pipeline()

    

    # Create color camera
    camRgb = pipeline.createColorCamera()
    camRgb.setPreviewSize(300, 300)
    camRgb.setInterleaved(False)

    # Create mono cameras for StereoDepth
    left = pipeline.create(dai.node.MonoCamera)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)

    right = pipeline.create(dai.node.MonoCamera)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # StereoDepth
    stereo = pipeline.createStereoDepth()
    stereo.initialConfig.setConfidenceThreshold(130)
    stereo.initialConfig.setLeftRightCheckThreshold(150)
    # Enable median filtering
    stereo.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
    
    # Setting node configs
    lrcheck = True
    subpixel = True

    stereo.setLeftRightCheck(lrcheck)
    stereo.setSubpixel(subpixel)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

    left.out.link(stereo.left)
    right.out.link(stereo.right)

    spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)
    spatialLocationCalculator.setWaitForConfigInput(False)

    # Link StereoDepth to spatialLocationCalculator
    stereo.depth.link(spatialLocationCalculator.inputDepth)
    # Set initial config for the spatialLocationCalculator
    config = dai.SpatialLocationCalculatorConfigData()
    config.depthThresholds.lowerThreshold = 200
    config.depthThresholds.upperThreshold = 15000
    config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MIN

    
    # Create NxM roi grid
    N = 5 # Number of cells in x direction
    M = 5 # Number of cells in y direction
    safty_margin = 0.4 # add a safty margin to the roi on the edges(in percent) of total cell size
    for n in range(N):
        for m in range(M):
            # calulate safty margin 
            left_margin = safty_margin*(1/N) if n == 0 else 0
            right_margin = safty_margin*(1/N) if n == N-1 else 0
            upper_margin = safty_margin*(1/M) if m == 0 else 0
            lower_margin = safty_margin*(1/M) if m == M-1 else 0

            # Define ROI for each cell            
            topLeft = dai.Point2f((n%N)*1/N + left_margin, (m%M)*1/M + upper_margin)
            bottomRight = dai.Point2f(((n%N)+1)*1/N - right_margin, ((m%M)+1)*1/M - lower_margin)
            # print(f"topLeft: {topLeft}, bottomRight: {bottomRight}")
            config.roi = dai.Rect(topLeft, bottomRight)
            # Add the ROI 
            spatialLocationCalculator.initialConfig.addROI(config)
            
    


    # Send depth frames to the host
    xoutDepth = pipeline.createXLinkOut()
    xoutDepth.setStreamName("depth")
    spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)

    # Send spatialLocationCalculator data to the host through the XLink
    xoutSpatialData = pipeline.createXLinkOut()
    xoutSpatialData.setStreamName("spatialData")
    spatialLocationCalculator.out.link(xoutSpatialData.input)

    # Send spatialLocationCalculator data through the SPI
    spiOutSpatialData = pipeline.create(dai.node.SPIOut)
    spiOutSpatialData.setStreamName("spatialData")
    spiOutSpatialData.setBusId(0)
    spiOutSpatialData.input.setBlocking(False)
    spiOutSpatialData.input.setQueueSize(4)
    spatialLocationCalculator.out.link(spiOutSpatialData.input)

    return pipeline

