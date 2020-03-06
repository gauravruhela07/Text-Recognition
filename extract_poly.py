# Import relevant modules
import numpy as np
import cv2
import matplotlib.pyplot as plt


def check(topleft, bottomright, width, height):
    if len(topleft)==2:
        if topleft[0]<0:
            topleft[0]=0
        if topleft[1]<0:
            topleft[1]=0
    if len(bottomright)==2:
        if bottomright[0]>width:
            bottomright[0]=width
        if bottomright[1]>height:
            bottomright[1]=height
    return topleft, bottomright


def extract_roi(img, pts):

    pts = np.array(pts, dtype=np.int32)
    # print(pts)
    # Define points
    # pts = np.array([[44,126], [176, 126], [176, 168], [44,168]], dtype=np.int32)
    # print(pts1)

    ### Define image here
    # img = 255*np.ones((300, 700, 3), dtype=np.uint8)
    # img = cv2.imread('db_new/images (1).png')

    # Initialize mask
    mask = np.zeros((img.shape[0], img.shape[1]))

    # Create mask that defines the polygon of points
    cv2.fillConvexPoly(mask, pts, 1)
    mask = mask.astype(np.bool)

    # Create output image (untranslated)
    out = np.zeros_like(img)
    out[mask] = img[mask]

    # Find centroid of polygon
    (meanx, meany) = pts.mean(axis=0)

    # Find centre of image
    (cenx, ceny) = (img.shape[1]/2, img.shape[0]/2)

    # Make integer coordinates for each of the above
    (meanx, meany, cenx, ceny) = np.floor([meanx, meany, cenx, ceny]).astype(np.int32)

    # Calculate final offset to translate source pixels to centre of image
    (offsetx, offsety) = (-meanx + cenx, -meany + ceny)

    # Define remapping coordinates
    (mx, my) = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    ox = (mx - offsetx).astype(np.float32)
    oy = (my - offsety).astype(np.float32)

    # Translate the image to centre
    out_translate = cv2.remap(out, ox, oy, cv2.INTER_LINEAR)
    # out_translate = cv2.remap(img, ox, oy, cv2.INTER_LINEAR)

    # Determine top left and bottom right of translated image
    topleft = pts.min(axis=0) + [offsetx, offsety]
    bottomright = pts.max(axis=0) + [offsetx, offsety]

    topleft, bottomright = check(topleft, bottomright, img.shape[0], img.shape[1])
    roi = out_translate[topleft[1]:bottomright[1], topleft[0]:bottomright[0]]
    return roi
