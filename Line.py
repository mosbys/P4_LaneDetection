# Define a class to receive the characteristics of each line detection
import numpy as np
from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = np.zeros([10,720]) 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

    def addMeasurement(self,x_pixel,y_pixel,coeffpoly):
        # add last measuremnt values
        self.ally = y_pixel
        self.allx = x_pixel
        self.current_fit = coeffpoly
        
        
        
        

        self.CalcUpdate()

    def CalcUpdate(self):
        xm_per_pix = 3.7/700 
        if (self.recent_xfitted[9].sum()==0):
            #first assaignment . fill filter up with first value
            for iIndex in range(0,9):
                self.recent_xfitted[iIndex]=self.allx
            
        self.recent_xfitted[1:10]=self.recent_xfitted[0:-1]
        self.recent_xfitted[0]=self.allx
        self.bestx = self.recent_xfitted.mean(axis=0)
       
        self.radius_of_curvature=self.CalcCurvature()
        self.line_base_pos = ((self.bestx[0])-1280/2)*(xm_per_pix)
        

    def CalcCurvature(self):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = (30/720) # meters per pixel in y dimension
        xm_per_pix = (3.7/700) # meters per pixel in x dimension

        ploty = self.ally
        fit_cr=self.current_fit
        y_eval = np.max(ploty)
        
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, self.bestx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        
    
        return curverad