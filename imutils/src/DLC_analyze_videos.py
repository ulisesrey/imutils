import argparse
import os
os.environ["DLClight"]="True"
import deeplabcut
import sys
import time  # Import the time module


def main(arg_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_config_file", required=True, help="path to config file")
    parser.add_argument("--videofile_path", required=True, help="path to the videofile")
    args = parser.parse_args(arg_list)

    print(args)
    path_config_file = args.path_config_file
    videofile_path = args.videofile_path
    print(path_config_file)
    VideoType = 'avi'

    print("Videofilepath: ", videofile_path)

    #although this code on the jupyternotebook works, i had an error while using the .py file. It went away with gputouse=None (although I was in a GPU env)
    deeplabcut.analyze_videos(path_config_file, videofile_path, videotype=VideoType, gputouse=None, save_as_csv=True)
    deeplabcut.filterpredictions(path_config_file, videofile_path, videotype=VideoType, save_as_csv=True, windowlength=3)

    # Wait for 1 minute (60 seconds) for server file system:
    print("Waiting for the server file system to be ready to create labeled videos")
    time.sleep(60)  # Wait for 60 seconds
    print("Resuming create labeled videos...")

    deeplabcut.create_labeled_video(path_config_file, videofile_path, videotype=VideoType, save_frames=False)

if __name__ == "__main__":
    main(sys.argv[1:])
