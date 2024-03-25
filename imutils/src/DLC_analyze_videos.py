import argparse
import os
os.environ["DLClight"]="True"
import deeplabcut
import sys


def main(arg_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(" --path_config_file", required=True, help="path to config file")
    parser.add_argument("--videofile_path", required=True, help="path to the videofile")
    #args = vars(ap.parse_args())
    args = parser.parse_args(arg_list)

    print(args)
    path_config_file = args.path_config_file
    videofile_path = args.videofile_path
    print(path_config_file)
    VideoType = 'avi'

    #don't edit these:
    #videos dont need to be on the config file: https://gitter.im/DeepLabCut/community?at=5e8f90a85d148a0460f7664a


    print("Videofilepath: ", videofile_path)

    #although this code on the jupyternotebook works, i had an error while using the .py file. It went away with gputouse=None (although I was in a GPU env)
    deeplabcut.analyze_videos(path_config_file, videofile_path, videotype=VideoType, gputouse=None, save_as_csv=True)
    deeplabcut.filterpredictions(path_config_file, videofile_path, videotype=VideoType, save_as_csv=True, windowlength=3)

if __name__ == "__main__":
    main(sys.argv[1:])
