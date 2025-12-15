import os
import pickle
from utils.helper_functions import load_yaml



dataset_root = load_yaml('config/configs.yml')['dataset_root']
output_dir = load_yaml('config/configs.yml')['output_dir']



def load_video_annot(video_annot):
    with open(video_annot, 'r') as file:
        clip_category_dct = {}

        for line in file:
            items = line.strip().split(' ')[:2]
            clip_dir = items[0].replace('.jpg', '')
            clip_category_dct[clip_dir] = items[1]

        return clip_category_dct



def load_player_annots(video_annot):

    with open(video_annot, 'r') as f:
        frame_annotations = {}

        for line in f:
            parts = line.strip().split()
            frame_image = parts[0]
            frame_id = os.path.splitext(frame_image)[0]

            players = []
            for i in range(2, len(parts), 5):
                if i + 4 < len(parts):
                    X, Y, W, H = map(int, parts[i:i+4])
                    action_class = parts[i + 4]
                    players.append({
                        'action_class': action_class,
                        'X': X,
                        'Y': Y,
                        'W': W,
                        'H': H
                    })

            frame_annotations[frame_id] = players

    return frame_annotations






    


def load_volleyball_dataset_for_the_middle_frame(videos_root):
    videos_dirs = os.listdir(videos_root)
    videos_dirs.sort()

    videos_annot = {}

    # Iterate on each video and for each video iterate on each clip
    for idx, video_dir in enumerate(videos_dirs):
        video_dir_path = os.path.join(videos_root, video_dir)

        if not os.path.isdir(video_dir_path):
            continue

        print(f'{idx + 1}/{len(videos_dirs)} - Processing Dir {video_dir_path}')

        video_annot = os.path.join(video_dir_path, 'annotations.txt')
        clip_category_dct = load_video_annot(video_annot)
        frame_boxes_dct = load_player_annots(video_annot)
        

        clips_dir = os.listdir(video_dir_path)
        clips_dir.sort()

        clip_annot = {}

        for clip_dir in clips_dir:
            clip_dir_path = os.path.join(video_dir_path, clip_dir)

            if not os.path.isdir(clip_dir_path):
                continue
            clip_annot[clip_dir] = {
                    'category': clip_category_dct[clip_dir],
                    'frame_boxes_dct': frame_boxes_dct[clip_dir]
                }

        videos_annot[video_dir] = clip_annot

    return videos_annot


def create_pkl_version():
    # You can use this function to create and save pkl version of the dataset
    videos_root = f'{dataset_root}/volleyball_/videos'

    videos_annot = load_volleyball_dataset_for_the_middle_frame(videos_root)

    with open(f'{output_dir}/annots_one_frame_per_clip.pkl', 'wb') as file:
        pickle.dump(videos_annot, file)






if __name__ == "__main__":

    create_pkl_version()