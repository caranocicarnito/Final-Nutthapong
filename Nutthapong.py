import cv2
import numpy as np
import os
import scipy

def feature_object_detection(template_img, template_gray, query_img, query_gray, min_match_number) :
    template_kpts, template_desc = sift.detectAndCompute(template_gray, None)
    query_kpts, query_desc = sift.detectAndCompute(query_gray, None)
    matches = bf.knnMatch(template_desc, query_desc, k=2)
    good_matches = list()
    good_matches_list = list()
    
img = cv2.imread("photo.JPG", cv2.IMREAD_GRAYSCALE)
cap = cv2.VideoCapture("left_output-1.avi")

sift = cv2.xfeatures2d.SIFT_create()
kp_image, desc_image = sift.detectAndCompute(img, None)


index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()
while True:
    _, frame = cap.read()
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
    matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
    matche = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            matche.append(m)
   

    if len(matche) > 6:
        scr_pts = np.float32([kp_image[m.queryIdx].pt for m in matche]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in matche]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(scr_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        h, w = img.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        homography = cv2.polylines(frame, [np.int32(dst)], True, (10, 255, 10), 3)
        cv2.imshow("Homography", homography)
    else:
        cv2.imshow("Homography", grayframe)
    key = cv2.waitKey(1)
    if cv2.waitKey(int(1000/24)) & 0xFF == ord('q') : 
        break

cap.release()
cv2.destroyAllWindows()
