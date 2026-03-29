import imghdr
import os
import cv2

def hist_difference(frames): # calculate histogram difference
    hist_diffs = []
    for i in range(len(frames) - 1):
        frame_a = frames[i]
        frame_b = frames[i + 1]
        
        if frame_a is not None and frame_b is not None:
            hist_a = cv2.calcHist([frame_a], [0], None, [256], [0, 256])
            hist_b = cv2.calcHist([frame_b], [0], None, [256], [0, 256])

            difference = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
            hist_diffs.append(difference)
        else:
            hist_diffs.append(0.0)  

    return hist_diffs

def select_top_k_key_frames(input_folder, output_folder, target_frame_count, k):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("Input folder contents:")
    for item in os.listdir(input_folder):
        print(item)

    for video_file in os.listdir(input_folder):
        video_file_path = os.path.join(input_folder, video_file)

        print(f"Checking video: {video_file_path}")

        if os.path.isfile(video_file_path) and is_video_file(video_file_path):
            print("File is a video file")
            video_id = os.path.splitext(video_file)[0]  # Extract video ID from the file name

            cap = cv2.VideoCapture(video_file_path)

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Ensure we have at least one frame
            if frame_count < 1:
                print("Video has no frames.")
                continue

            # Determine the step size for frame selection
            step_size = max(frame_count // target_frame_count, 1)

            # Reset the frame count to 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                            
            while len(frames) < target_frame_count:
                # Duplicate frames until we reach the target frame count
                frames.append(frames[-1])
            print(len(frames))

            differences = hist_difference(frames)

            # Select the frames with the highest differences
            selected_indices = sorted(range(len(differences)), key=lambda i: differences[i], reverse=True)[:k]

            # Adjust the selected indices to start from 0
            selected_indices = list(range(min(target_frame_count, len(frames))))

            for index, selected_index in enumerate(selected_indices):
                frame_path = os.path.join(output_folder, f'{video_id}_{index}.jpg')
                cv2.imwrite(frame_path, frames[selected_index])

            cap.release()
        else:
            print("File is not a video file")

def is_video_file(file_path):
    # Check if the file is a video file by verifying it's not an image file
    return imghdr.what(file_path) is None

select_top_k_key_frames('/Users/PRASITHA/OneDrive/Desktop/demo_v3/dataset', '/Users/PRASITHA/OneDrive/Desktop/demo_v3/final_extracted_frames', 45, 45)