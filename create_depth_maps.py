import cv2
import numpy as np
import torch
import os
import argparse

def create_depth_maps(npy_path, savepath, mode):

	if not os.path.exists(savepath):
		os.makedirs(savepath)
	
	left_frame_path = npy_path+'capture1/frames/'
	
	for left_cam_frame in os.listdir(left_frame_path):

		right_frame_path = npy_path+'capture2/frames/'
		
		if os.path.isfile(right_frame_path+left_cam_frame):

			L = np.load(left_frame_path+left_cam_frame)
			R = np.load(right_frame_path+left_cam_frame)
			imgL=cv2.cvtColor(L, cv2.COLOR_RGB2GRAY)
			imgR=cv2.cvtColor(R, cv2.COLOR_RGB2GRAY)

			if imgL.shape==imgR.shape:
				# SGBM Parameters -----------------
				window_size = 3        
				min_disp = 16
				num_disp = 112-min_disp
				 
				left_matcher = cv2.StereoSGBM_create(
				    minDisparity=min_disp,
				    numDisparities=num_disp,
				    blockSize=5,
				    P1=8*3*window_size**2,    
				    P2=32*3*window_size**2,
				    disp12MaxDiff=1,
				    uniquenessRatio=15,
				    speckleWindowSize=0,
				    speckleRange=2,
				    preFilterCap=63,
				    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
				)
				right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
				
				# FILTER Parameters
				lmbda = 80000
				sigma = 1.2
				visual_multiplier = 1.0
				 
				wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
				wls_filter.setLambda(lmbda)
				wls_filter.setSigmaColor(sigma)

				displ = left_matcher.compute(imgL, imgR)
				dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
				displ = np.int16(displ)
				dispr = np.int16(dispr)
				filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

				filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
				filteredImg = np.uint8(filteredImg)

				np.save(savepath+left_cam_frame, filteredImg)
			
			else:
				continue
		else:
			continue
					
def main(args):
	
	if args.operation==0:
		base_path = '/storage/soumava/JIGSAWS/Knot_Tying/npy_files/'
	elif args.operation==1:
		base_path = '/storage/soumava/JIGSAWS/Needle_Passing/npy_files/'
	elif args.operation==2:
		base_path = '/storage/soumava/JIGSAWS/Suturing/npy_files/'

	train_npy = base_path+'train/'
	val_npy = base_path+'val/'
	test_npy = base_path+'test/'

	train_savepath = base_path+'train/depth_maps/'
	val_savepath = base_path+'val/depth_maps/'
	test_savepath = base_path+'test/depth_maps/'

	print('Processing and saving train data')
	create_depth_maps(train_npy, train_savepath, 'train')
	print('Done\n')
	
	print('Processing and saving val data')
	create_depth_maps(val_npy, val_savepath, 'val')
	print('Done\n')

	print('Processing and saving test data')
	create_depth_maps(test_npy, test_savepath, 'test')
	print('Done')

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Regression Models for Surgical Videos.")
	parser.add_argument('-op', '--operation', help="choose between 0(knot-tying), 1(needle-passing), 2(suturing)", default=0, type=int)
	
	args = parser.parse_args()
	main(args)
