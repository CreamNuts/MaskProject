import os, argparse
import cv2, dlib, imutils
import numpy as np
  
def create_mask(img_dir, mask_choice=2, mask_color=3):
    assert os.path.isfile(img_dir), "Wrong image directory"
    img = cv2.imread(img_dir)
    detector = dlib.get_frontal_face_detector()
    faces = detector(img, 1) #1은 사진 확대 계수, 크면 얼굴 더 잘 찾을 수 있음
    assert len(faces) == 1, "This image doesn't have face"
    black = np.zeros(shape=[img.shape[0], img.shape[1], 1], dtype=np.uint8)

    p = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(p)
    
    for face in faces:
        landmarks = predictor(img, face) #위에 assert걸어서 사실 for문 필요 없긴 함
    
    points = []
    for i in range(1, 16):
        point = [landmarks.part(i).x, landmarks.part(i).y]
        points.append(point)
    # Coordinates for the additional 3 points for wide, high coverage mask - in sequence
    mask_a = [((landmarks.part(42).x), (landmarks.part(15).y)),
                ((landmarks.part(27).x), (landmarks.part(27).y)),
                ((landmarks.part(39).x), (landmarks.part(1).y))]

    # Coordinates for the additional point for wide, medium coverage mask - in sequence
    mask_c = [((landmarks.part(29).x), (landmarks.part(29).y))]

    # Coordinates for the additional 5 points for wide, low coverage mask (lower nose points) - in sequence
    mask_e = [((landmarks.part(35).x), (landmarks.part(35).y)),
                ((landmarks.part(34).x), (landmarks.part(34).y)),
                ((landmarks.part(33).x), (landmarks.part(33).y)),
                ((landmarks.part(32).x), (landmarks.part(32).y)),
                ((landmarks.part(31).x), (landmarks.part(31).y))]

    fmask_a = np.array(points + mask_a, dtype=np.int32)
    fmask_c = np.array(points + mask_c, dtype=np.int32)
    fmask_e = np.array(points + mask_e, dtype=np.int32)

    mask_type = {1: fmask_a, 2: fmask_c, 3: fmask_e}

    color_blue = (239,207,137)
    color_black = (0, 0, 0)
    color_white = (255,255,255)

    color_type = {1: color_blue, 2: color_black, 3: color_white}
    # change parameter [mask_type] and color_type for various combination
    outline = cv2.polylines(img, [mask_type[mask_choice]], True, color_type[mask_color], thickness=2, lineType=cv2.LINE_8)
    #cv2.imwrite(img_dir[:-4]+"_outline.jpg", outline)
    mask = cv2.fillPoly(outline, [mask_type[mask_choice]], color_type[mask_color], lineType=cv2.LINE_AA)
    outline = cv2.polylines(black, [mask_type[mask_choice]], True, 255, thickness=2, lineType=cv2.LINE_8)
    only_mask = cv2.fillPoly(outline, [mask_type[mask_choice]], 255, lineType=cv2.LINE_AA)

    outputNameofMask = img_dir[:-4]+"_mask.jpg"
    outputNameofOnlymask = img_dir[:-4]+"_maskgt.jpg"
    cv2.imwrite(outputNameofMask, mask)    
    cv2.imwrite(outputNameofOnlymask, only_mask)    
    #print(f"Saving output image to {outputNameofMask} and {outputNameofOnlymask}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wear a face mask in the given picture')
    parser.add_argument('img_dir', type=str, help='Image Directory')
    parser.add_argument('--mask_type', dest='mask', type=int, default=2, choices=[1, 2, 3], 
    help='Mask Type, 1: High coverage, 2: Medium coverage, 3: Low converage'
    )
    parser.add_argument('--color_type', dest='color', type=int, default=3, choices=[1, 2, 3], 
    help='Mask Color Type, 1: Blue, 2: Black, 3: White'
    )
    args = parser.parse_args()
    create_mask(args.img_dir, mask_choice=args.mask, mask_color=args.color)