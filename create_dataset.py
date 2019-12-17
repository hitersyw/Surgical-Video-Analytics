import cv2
import numpy as np
import torch
import os
import argparse

def extract_Xy(video_path, label_path):
	
	frame_count=0
	video_count=0
	
	for video in os.listdir(video_path):

		if video[-5]=='1':

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
						image = cv2.resize(image, (256,256), interpolation=cv2.INTER_CUBIC)
						image = np.expand_dims(image, 0)
						
						if frame_count:
							frame_array = np.concatenate((frame_array, image), axis=0)
						else:
							frame_array = image
						frame_count+=1
						i+=1
					
					else:
						break
				else:
					break
			
			cap.release()

			num_kin_frames = min(i, kin_var.shape[0])
			
			if video_count:
				kin_labels = np.concatenate((kin_labels, kin_var[:num_kin_frames, 38:]), axis=0)
			else:
				kin_labels = kin_var[:num_kin_frames, 38:] # Only slave side kinematic variables, also some of the last frames don't have corr video data
			
			print('Data shape:{}'.format(frame_array.shape))
			print('Labels shape:{}'.format(kin_labels.shape))

			video_count+=1
			
	data = torch.from_numpy(np.transpose(frame_array, (0,3,2,1)))
	data = data.type(torch.ByteTensor)
	
	print('Final data shape:{}'.format(data.shape))
	print('Final labels shape:{}'.format(kin_labels.shape))
	
	return data, kin_labels

def main(args):
	
	if args.operation==0:
		base_path = '../JIGSAWS/Knot_Tying/'
	elif args.operation==1:
		base_path = '../JIGSAWS/Needle_Passing/'
	elif args.operation==2:
		base_path = '../JIGSAWS/Suturing/'

	train_videos = base_path+'video/train/'
	val_videos = base_path+'video/val/'
	test_videos = base_path+'video/test/'

	label_path = base_path+'kinematics/AllGestures/'

	savepath = base_path+'data/'

	if not os.path.exists(savepath):
		os.makedirs(savepath)

	print('Processing and saving train data')
	train_data, train_labels = extract_Xy(train_videos, label_path)
	torch.save(train_data, savepath+'train_256.pth')
	np.save(savepath+'train_kin.npy', train_labels)
	print('Done\n')
	
	print('Processing and saving val data')
	val_data, val_labels = extract_Xy(val_videos, label_path)
	torch.save(val_data, savepath+'val_256.pth')
	np.save(savepath+'val_kin.npy', val_labels)
	print('Done\n')

	print('Processing and saving test data')
	test_data, test_labels = extract_Xy(test_videos, label_path)
	torch.save(test_data, savepath+'test_256.pth')
	np.save(savepath+'test_kin.npy', test_labels)
	print('Done')

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Regression Models for Surgical Videos.")
	parser.add_argument('-op', '--operation', help="choose between 0(knot-tying), 1(needle-passing), 2(suturing)", default=0, type=int)
	
	args = parser.parse_args()
	main(args)
