import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import time
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from google.colab.patches import cv2_imshow

import pickle

DEVICE = 'cuda'

def image_resize(image, new_size):
  # Function that resizes an 3D array (image) into a new shape defined by the new_size tuple
  # Inputs:
  # image: A 3D numpy array containing image data
  # new_size: A 2 entry tuple giving (height, width) 
  # Outputs:
  # new_image: resized image (note that the aspect ratio is not retained)

  # Convert to Image object then resize
  PIL_image = Image.fromarray(image)
  new_im = PIL_image.resize(new_size[::-1]) # We need to reverse the size tuple as PIL expects (width, height)
  
  # return after converting to array
  return np.asarray(new_im)


def viz(img, flo, i):
  # Convert from rgb to bgr
  img = img[0].permute(1,2,0).cpu().numpy()
  flo = flo[0].permute(1,2,0).cpu().numpy()
  
  # map flow to rgb image
  flo = flow_viz.flow_to_image(flo)
  img_flo = np.concatenate([img, flo], axis=0)
  
  cv2_imshow(img_flo[:, :, [2,1,0]]/255.0)
  filename = '{}.jpg'.format(i)
  cv2.imwrite(filename, img_flo[:, :, [2,1,0]])
  cv2.waitKey()

def write_vid(args):

  # Setup model
  model = torch.nn.DataParallel(RAFT(args))
  model.load_state_dict(torch.load(args.model))

  model = model.module
  model.to(DEVICE)
  model.eval()
  if args.write_location == None: raise ValueError('You must provide a place to write the video.')

  # Setup file write location from arguments
  write_location = os.path.join(args.write_location, 'processedVideo.mp4')

  with torch.no_grad():
    # Get video and read to cv2 object
    vid = cv2.VideoCapture(args.path)

    baseVideoInformation = {}
    # Retrieve data to write out later
    baseVideoInformation["Channels"] = 3
    baseVideoInformation["FPS"] = int(vid.get(cv2.CAP_PROP_FPS))
    baseVideoInformation["Width"] = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    baseVideoInformation["Height"] = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    baseVideoInformation["Frames"] = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # Count frames read
    count = 0

    # Create video writer object
    fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
    out = cv2.VideoWriter(write_location, fourcc, baseVideoInformation["FPS"], (1024, 880))

    # Time execution option
    if (args.time == "True" or args.time == "true"):
      tic = time.perf_counter()

    
    while(vid.isOpened()):

      ret, frame1 = vid.read()
      count+=1
      if not ret: # Faliure to get the next frame
        print('Error on first frame.')
        break

      # Read and prepare the first frame ( convert BGR to RGB then resize, and reframe)
      image1 = image_resize(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB), (440, 1024)) # This gives [h], [w], [ch]

      image1 = np.transpose(image1, axes=[2, 0, 1])[np.newaxis, :, :, :] # We need to reframe to [ex], [ch], [h], [w]

      image1 = torch.from_numpy(image1).to(DEVICE)
      
      # Initialise lists
      if (args.save_data):
        flow_map = []
        img_map = []

      # Main processing loop
      while(1):
        ret, frame2 = vid.read()
        count+=1
        if not ret: # Faliure to get the next frame
          print('Finished trying to retrieve frame {}, out of {}.'.format(count, baseVideoInformation["Frames"]))
          break

        if not (count%100): print('Processing frame {}, out of {}.'.format(count, baseVideoInformation["Frames"]))

        # Read and process the following frames
        image2 = image_resize(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB), (440, 1024))

        image2 = np.transpose(image2, axes=[2, 0, 1])[np.newaxis, :, :, :]

        image2 = torch.from_numpy(image2).to(DEVICE)

        # Pass a pair of images(to calculate flow from) to the model
        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

        ### Write to file ###
        image1 = image1[0].permute(1,2,0).cpu().numpy()
        flow_up = flow_up[0].permute(1,2,0).cpu().numpy()
        
        # Append data to a list to save later
        if (args.save_data):
          flow_map.append(flow_up)
          img_map.append(image1)
        
        # map flow to rgb image
        flo = flow_viz.flow_to_image(flow_up, rad_max=args.scale)
        img_flo = np.concatenate([image1, flo], axis=0)
        
        # Reframe BGR back to RGB
        img_write = img_flo[:, :, [2,1,0]]
        out.write(img_write)

        # Write image2 to image1 
        image1 = image2

      # Time execution option
      if (args.time == "True" or args.time == "true"):
        toc = time.perf_counter()
        print(f"Processing completed in {toc - tic:0.4f} seconds")

      out.release()
      cv2.destroyAllWindows()
      print("Video resources relieved")
      
      # Convert data lists to pickle and save 
      if (args.save_data):
        flow_map = np.asarray(flow_map)
        img_map = np.asarray(img_map)
      
        # Extract relevant filename
        filename = args.path.split(".")[0].split("/")[-1]

        flow_name = "{}_flow_map.pkl".format(filename)
        img_name = "{}_img_map.pkl".format(filename)

        # save
        with open(flow_name, 'wb') as f:
          pickle.dump(flow_map, f)
        
        with open(img_name, 'wb') as f:
          pickle.dump(img_map, f)

      break

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', help="restore checkpoint")
  parser.add_argument('--time', help="True to time execution")
  parser.add_argument('--path', help="Video for evaluation")
  parser.add_argument('--write_location', help="location to write output to")
  parser.add_argument('--small', action='store_true', help='use small model')
  parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
  parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
  parser.add_argument('--manual_scale', dest='scale', type=int, help='Manual scaling number (leave blank for auto scaling)', default=None)
  parser.add_argument('--save_data', action='store_true', help='toggle to store img and flow maps')
  args = parser.parse_args()

  # Example argument !python3 /content/RAFT/convert_video.py --model=models/raft-sintel.pth --path=data/src_finch_174.mp4 --write_location=./
  write_vid(args)
