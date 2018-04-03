
from __future__ import print_function
import cv2
import os
import time

def save_video_images_to_folder(video_dir, save_dir):
    # traet input folder
    assert os.path.exists(save_dir) == False, "This save directory already exists"
    os.makedirs(save_dir)

    # read video
    vidcap = cv2.VideoCapture(video_dir)
    count = 0
    start_time = time.time()
    while True:
        success,image = vidcap.read()
        if not success:
            break
        print('Read a new frame %-4d, time used: %8.2fs \r' % (count, time.time()-start_time), end="")
        cv2.imwrite(os.path.join(save_dir, 'frame{}.jpg'.format(count)), image)     # save frame as JPEG file
        count += 1

def generate_video(video_name, path_format, image_format):
    #image_folder = 'output'
    #images = [img for img in os.listdir(image_dir) if img.endswith(".jpg")]
    images =dict()
    count =0
    i=0
    while True:
        i = i+1
        new_path = path_format.format(i)

        if os.path.exists(new_path) == True:
             for element in os.listdir(new_path):
                if(str(element).endswith('jpg')):
                    count = count + 1
                    images[element] = os.path.join(new_path, element)
        else:
            break
    print(count)
    print(image_format.format(1))
    frame = cv2.imread(images[image_format.format(1)])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    video = cv2.VideoWriter(video_name, fourcc, 60.0, (width,height))
    start_time = time.time()
    i = 1
    j = 1
    while i < count+1:
        if image_format.format(j) in list(images.keys()):
            video.write(cv2.imread(images[image_format.format(j)]))
            print('(%d/%d)write a new frame, time used: %5.2fs \r' % (i, count, time.time()-start_time), end="")
            i = i+1
        j=j+1
    print('\n %s created.' % video_name)
    cv2.destroyAllWindows()
    video.release()

def play_video(video_dir):
    cap = cv2.VideoCapture(video_dir)
    paused = False
    delay = {True: 0, False: 1}
    while(True):
        ret, frame = cap.read()
        if not ret:
            cap = cv2.VideoCapture(video_dir)
            ret, frame = cap.read()
        cv2.putText(frame, 'Press \'q\' to stop.', (20, 20), 0, 0.5, (0, 0, 255))
        cv2.putText(frame, 'Press \'p\' to pause.', (20, 40), 0, 0.5, (0, 0, 255))
        cv2.imshow('frame',frame)
        time.sleep(0.1)    #10fps
        key = cv2.waitKey(delay[paused])
        if key & 255 == ord('p'):
            paused = not paused

        if key & 255 == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
#    video_dir = './test2.mp4'
#    save_dir = './video_image_set/test2_image'
#    save_video_images_to_folder(video_dir, save_dir)
	path_format= '/scratch/js5991/video/video_image_set/test1_part{}_detected'
	image_format = 'frame{}_detected.jpg'

	generate_video('test1_demo.mp4', path_format, image_format)
