import os

import cv2


def convert(path_to_video):
    # Open the input video
    cap = cv2.VideoCapture(path_to_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join('videos', "converted.mp4"), fourcc, fps, (1920, 1080))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Resize the frame
        resized_frame = cv2.resize(frame, (1920, 1080))
        out.write(resized_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    convert(os.path.join("videos", "3760967-uhd_3840_2160_25fps.mp4"))
