#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:27:05 2023

@author: yuri
"""

import numpy as np


class UWBMeasurement(): 
    def __init__(self, maxRange): 
        self.maxRange = maxRange 
        self.noiseModels = ["Guassian", "HeavyTailGaussian"]
        self.noiseModel = "Guassian"
    def setNoiseModel(self, noiseModel: str = "Gaussian"): 
        if noiseModel in self.noiseModels: 
            self.noiseModel = noiseModel
        else: 
            print(f"{noiseModel} is not implemnented as a noise model.")
            
            
    def UWBMeasurment(self, distance): 
        return eval("self.run"+self.noiseModel+"NoiseModel("+str(distance)+")")
        
    #--------------------------------------------------------------------------
    # ---- Guassian Noise                                                   ---
    #--------------------------------------------------------------------------
    def setGuassianNoiseParameters(self, sigmaDistance, muDistance):
        self.guassian_sigmaDistance = sigmaDistance 
        self.guassian_muDistance = muDistance 
        
    def runGuassianNoiseModel(self, distance): 
        return distance + self.guassian_muDistance + np.random.randn(1)*self.guassian_sigmaDistance
    
    #--------------------------------------------------------------------------
    # ---- Heavy Tail Gaussian                                              ---
    #--------------------------------------------------------------------------
    def setHeavyTailGaussianNoiseParameters(self, sigmaDistance, muDistance):
        print("Please implement this function.")
    
    #--------------------------------------------------------------------------
    # ---- Distance effect                                                  ---
    #--------------------------------------------------------------------------
    
    