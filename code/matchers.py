import cv2
import numpy as np
class Matchers:
    def __init__(self):
        self.orb = cv2.ORB_create(5000)  # max 5000 keypoints
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match(self, img1, img2, direction=None):
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and descriptors
        kp1, des1 = self.orb.detectAndCompute(gray1, None)
        kp2, des2 = self.orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            return None

        # KNN match
        matches = self.bf.knnMatch(des1, des2, k=2)

        # Apply ratio test (Lowe's)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append((m.queryIdx, m.trainIdx))

        if len(good_matches) < 4:
            return None

        # Extract matched points
        pts1 = np.float32([kp1[i].pt for i, _ in good_matches])
        pts2 = np.float32([kp2[j].pt for _, j in good_matches])

        # Compute homography with RANSAC
        H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

        # Validate H
        if H is None or np.any(np.abs(H) > 1e4):
            return None

        return H
