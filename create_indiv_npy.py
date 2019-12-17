import cv2
import numpy as np
import torch
import os
import argparse

def extract_Xy(video_path, label_path, savepath, mode):
	
	video_count=0
	
	for video in os.listdir(video_path):

		print('Video#:{}'.format(video_count+1))

		label_file_name = label_path+video[:-13]+'.txt'
		kin_var = np.loadtxt(label_file_name)
		cap = cv2.VideoCapture(video_path+video)
		
		i=0
		
		while(1):
			
			ret,frame = cap.read()

			if (i+1)<=kin_var.shape[0]:
			
				if ret == True:

					if i:
						current = frame
						diff = current-prev
						prev = current
						avg_diff = np.mean(diff, axis=2)
						total_pixel_diff = np.mean(avg_diff, axis=None)
						if total_pixel_diff==0.0:
							print('Video freezing starts...not processing any further frames')
							break
					else:
						prev = frame
					
					image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
					# if mode=='train':
					# 	image = cv2.resize(image, (256,256), interpolation=cv2.INTER_CUBIC)
					# else:
					# 	image = cv2.resize(image, (224,224), interpolation=cv2.INTER_CUBIC)

					frame_label = kin_var[i, 38:] # Only slave side kinematic variables, also some of the last frames don't have corr video data

					np.save(savepath+mode+'/capture'+video[-5]+'/frames/'+video[-17:-12]+str(i+1)+'.npy', image)
					np.save(savepath+mode+'/capture'+video[-5]+'/labels/'+video[-17:-12]+str(i+1)+'.npy', frame_label)
					i+=1
				
				else:
					break
			else:
				break
		
		print('#Frames processed={}'.format(i))
		
		cap.release()
		
		video_count+=1

def main(args):
	
	if args.operation==0:
		base_path = '/storage/soumava/JIGSAWS/Knot_Tying/'
	elif args.operation==1:
		base_path = '/storage/soumava/JIGSAWS/Needle_Passing/'
	elif args.operation==2:
		base_path = '/storage/soumava/JIGSAWS/Suturing/'

	train_videos = base_path+'video/train/'
	val_videos = base_path+'video/val/'
	test_videos = base_path+'video/test/'

	label_path = base_path+'kinematics/AllGestures/'

	savepath = base_path+'npy_files/'

	if not os.path.exists(savepath):
		os.makedirs(savepath)

	print('Processing and saving train data')
	extract_Xy(train_videos, label_path, savepath, 'train')
	print('Done\n')
	
	print('Processing and saving val data')
	extract_Xy(val_videos, label_path, savepath, 'val')
	print('Done\n')

	print('Processing and saving test data')
	extract_Xy(test_videos, label_path, savepath, 'test')
	print('Done')

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Regression Models for Surgical Videos.")
	parser.add_argument('-op', '--operation', help="choose between 0(knot-tying), 1(needle-passing), 2(suturing)", default=0, type=int)
	
	args = parser.parse_args()
	main(args)
