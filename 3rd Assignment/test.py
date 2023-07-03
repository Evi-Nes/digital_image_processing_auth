import numpy as np
import cv2

# load the images
image1 = cv2.imread('im1.png')
image2 = cv2.imread('im2.png')

#img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initiate SIFT detector
sift = cv2.SIFT_create()
# img1 = np.load('img1.npy', allow_pickle=True).item()
# keypoint1, descriptors1 = img1["corners"], img1["descriptor"]
# img2 = np.load('img2.npy', allow_pickle=True).item()
# keypoint2, descriptors2 = img2["corners"], img2["descriptor"]
# find the keypoints and descriptors with SIFT
keypoint1, descriptors1 = sift.detectAndCompute(image1, None)
keypoint2, descriptors2 = sift.detectAndCompute(image2, None)
# keypoint1 = list(keypoint1)
# keypoint2 = list(keypoint2)

# keypoints1 = [cv2.KeyPoint(x=kp[0], y=kp[1], _size=1) for kp in keypoint1]
# keypoints2 = [cv2.KeyPoint(x=kp[0], y=kp[1], _size=1) for kp in keypoint2]

descriptors1 = np.array(descriptors1, dtype=np.float32)
descriptors2 = np.array(descriptors2, dtype=np.float32)

# finding nearest match with KNN algorithm
index_params = dict(algorithm=0, trees=20)
search_params = dict(checks=150) # or pass empty dictionary

# Initialize the FlannBasedMatcher
flann = cv2.FlannBasedMatcher(index_params, search_params)

Matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Need to draw only good matches, so create a mask
good_matches = [[0, 0] for i in range(len(Matches))]

# Ratio Test
for i, (m, n) in enumerate(Matches):
	if m.distance < 0.05*n.distance:
		good_matches[i] = [1, 0]

# Draw the matches using drawMatchesKnn()
Matched = cv2.drawMatchesKnn(image1, keypoint1, image2, keypoint2, Matches, outImg=None, matchColor=(0, 155, 0), singlePointColor=(0, 255, 255), matchesMask=good_matches, flags=0)

# Displaying the image
cv2.imwrite('Match.jpg', Matched)