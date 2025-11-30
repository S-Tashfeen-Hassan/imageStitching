# FILE: matchers.py
import cv2
import numpy as np

class Matchers:
    def __init__(self):
        self.detector_name = None
        try:
            self.detector = cv2.SIFT_create()
            self.detector_name = 'sift'
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        except Exception:
            self.detector = cv2.ORB_create(5000)
            self.detector_name = 'orb'
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match(self, img1, img2, ratio_thresh=0.75, reproj_thresh=5.0):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)
        if des1 is None or des2 is None:
            return None
        try:
            knn = self.matcher.knnMatch(des1, des2, k=2)
        except Exception:
            if des1.dtype != np.float32:
                des1_ = des1.astype(np.float32)
                des2_ = des2.astype(np.float32)
                knn = self.matcher.knnMatch(des1_, des2_, k=2)
            else:
                raise
        good = []
        for m_n in knn:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < ratio_thresh * n.distance:
                good.append(m)
        if len(good) < 4:
            return None
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,2)
        H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, reproj_thresh)
        if H is None or np.any(np.isnan(H)) or np.any(np.isinf(H)) or np.linalg.cond(H) > 1e12:
            return None
        return H
