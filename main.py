import cv2
import utils

img = [0, 0]
print("Please input the two images' locations and finish with 'Enter'")
img[0] = '5-1.jpg'
img[1] = '5-2.jpg'

if img[0] and img[1]:
    image1 = cv2.imread(img[0])
    image2 = cv2.imread(img[1])
    ImgEnhance = utils.DataEnhance()
    # image1, image2 = ImgEnhance.BrightnessNormalization(image1,image2)
    stitch_match = utils.FindKeyPointsAndMatching()
    kp1, kp2 = stitch_match.get_key_points(img1=image1, img2=image2)
    homo_matrix = stitch_match.match(kp1, kp2, 'brute')
    stitch_merge = utils.PasteTwoImages()
    merge_image = stitch_merge(image1, image2, homo_matrix)
    cv2.imwrite('2-output.JPG', merge_image)
    # cv2.imshow('output.JPG', merge_image)
    # if cv2.waitKey() == 27:
    #     cv2.destroyAllWindows()

    print('\n=======>Output saved!')
else:
    print('Please input images locations in right way.')
