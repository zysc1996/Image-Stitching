import cv2
import numpy as np

class DataEnhance:
    def __init__(self):
        pass

    def BrightnessNormalization(self, img1, img2):
        img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img1_brt = np.mean(img1gray)
        img2_brt = np.mean(img2gray)
        bias = img1_brt - img2_brt
        for i in range(img1gray.shape[0]):
            for j in range(img1gray.shape[1]):
                img1gray[i][j] -= bias 
                if img1gray[i][j] < 0:
                    img1gray[i][j] = 0
                elif img1gray[i][j] > 255:
                    img1gray[i][j] = 255
        img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2BGRA)
        img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)
        return img1gray, img2gray
    
    def HistogramEqualization(self, img):
        (b, g, r) = cv2.split(img)
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        result = cv2.merge((bH, gH, rH))
        return result

class FindKeyPointsAndMatching:
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.brute = cv2.BFMatcher()
        self.FLANN_INDEX_KDTREE = 1
        self.index_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=5)
        self.search_params = dict(checks=50) 
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)

    def get_key_points(self, img1, img2):
        g_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        kp1, kp2 = {}, {}
        print('=======>Detecting key points!')
        kp1['kp'], kp1['des'] = self.sift.detectAndCompute(g_img1, None)
        kp2['kp'], kp2['des'] = self.sift.detectAndCompute(g_img2, None)
        return kp1, kp2

    def match(self, kp1, kp2, MatchMethod = 'brute'):
        print('=======>Matching key points!')
        if MatchMethod == 'brute':
            matches = self.brute.knnMatch(kp1['des'], kp2['des'], k=2)
        elif MatchMethod == 'flann':
            matches = self.flann.knnMatch(kp1['des'], kp2['des'], k=2)
        good_matches = []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good_matches.append((m.trainIdx, m.queryIdx))
        if len(good_matches) > 4:
            key_points1 = kp1['kp']
            key_points2 = kp2['kp']
            matched_kp1 = np.float32(
                [key_points1[i].pt for (_, i) in good_matches]
            )
            matched_kp2 = np.float32(
                [key_points2[i].pt for (i, _) in good_matches]
            )

            print('=======>Random sampling and computing the homography matrix!')
            homo_matrix, _ = cv2.findHomography(matched_kp1, matched_kp2, cv2.RANSAC, 4)
            return homo_matrix
        else:
            return None


class PasteTwoImages:
    def __init__(self):
        pass

    def __call__(self, img1, img2, homo_matrix):
        h1, w1 = img1.shape[0], img1.shape[1]
        h2, w2 = img2.shape[0], img2.shape[1]
        rect1 = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32).reshape((4, 1, 2))
        rect2 = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32).reshape((4, 1, 2))
        trans_rect1 = cv2.perspectiveTransform(rect1, homo_matrix)
        total_rect = np.concatenate((rect2, trans_rect1), axis=0)
        min_x, min_y = np.int32(total_rect.min(axis=0).ravel())
        max_x, max_y = np.int32(total_rect.max(axis=0).ravel())
        shift_to_zero_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
        trans_img1 = cv2.warpPerspective(img1, shift_to_zero_matrix.dot(homo_matrix), (max_x - min_x, max_y - min_y))
        trans_img1[-min_y:h2 - min_y, -min_x:w2 - min_x] = img2
        return trans_img1
