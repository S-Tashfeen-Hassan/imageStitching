import os
import cv2
import numpy as np
import argparse
import sys
from matchers import Matchers


def resolve_paths(list_file):
    base_dir = os.path.dirname(os.path.abspath(list_file))
    paths = []
    with open(list_file, 'r') as f:
        for line in f:
            p = line.strip()
            if not p:
                continue
            if not os.path.isabs(p):
                p = os.path.normpath(os.path.join(base_dir, p))
            paths.append(p)
    return paths


def load_images(paths, resize=None):
    imgs = []
    failed = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            failed.append(p)
            continue
        if resize is not None and resize > 0 and resize != 1.0:
            w = int(img.shape[1] * resize)
            h = int(img.shape[0] * resize)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        imgs.append(img)
    return imgs, failed


class Stitcher:
    def __init__(self, images, matcher: Matchers):
        assert len(images) >= 2, "Need at least two images to stitch"
        self.images = images
        self.matcher = matcher

    def compute_pairwise_homographies(self):
        Hs = []
        for i in range(len(self.images) - 1):
            H = self.matcher.match(self.images[i], self.images[i+1])
            Hs.append(H)
        return Hs

    def accumulate_to_center(self, Hs):
        n = len(self.images)
        center = n // 2
        transforms = [None] * n
        transforms[center] = np.eye(3)

        for i in range(center - 1, -1, -1):
            H_forward = Hs[i]
            if H_forward is None:
                transforms[i] = None
                continue
            try:
                H_inv = np.linalg.inv(H_forward)
            except np.linalg.LinAlgError:
                transforms[i] = None
                continue
            if transforms[i+1] is None:
                transforms[i] = None
            else:
                transforms[i] = transforms[i+1] @ H_inv

        for i in range(center + 1, n):
            H = Hs[i-1]
            if H is None or transforms[i-1] is None:
                transforms[i] = None
                continue
            transforms[i] = transforms[i-1] @ H

        return transforms

    def corners_after_transform(self, img, H):
        h, w = img.shape[:2]
        corners = np.array([[0,0,1],[w,0,1],[w,h,1],[0,h,1]]).T
        warped = H @ corners
        warped = warped / warped[2:3,:]
        xs = warped[0,:]
        ys = warped[1,:]
        return xs.min(), ys.min(), xs.max(), ys.max()

    def make_canvas(self, transforms):
        mins = [np.inf, np.inf]
        maxs = [-np.inf, -np.inf]
        for img, T in zip(self.images, transforms):
            if T is None:
                continue
            x1,y1,x2,y2 = self.corners_after_transform(img, T)
            mins[0] = min(mins[0], x1)
            mins[1] = min(mins[1], y1)
            maxs[0] = max(maxs[0], x2)
            maxs[1] = max(maxs[1], y2)

        margin = 10
        min_x, min_y = int(np.floor(mins[0])) - margin, int(np.floor(mins[1])) - margin
        max_x, max_y = int(np.ceil(maxs[0])) + margin, int(np.ceil(maxs[1])) + margin

        width = max_x - min_x
        height = max_y - min_y

        offset = np.array([[1,0,-min_x],[0,1,-min_y],[0,0,1]])
        return width, height, offset

    def warp_and_blend(self, transforms, canvas_size, offset):
        canvas_w, canvas_h = canvas_size
        acc = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        weight_sum = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        for img, T in zip(self.images, transforms):
            if T is None:
                print("Skipping an image because transform is None")
                continue
            H = offset @ T
            warped = cv2.warpPerspective(img, H, (canvas_w, canvas_h))
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            mask = (gray > 0).astype(np.uint8) * 255
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
            if dist.max() > 0:
                w = dist / (dist.max())
            else:
                w = mask.astype(np.float32) / 255.0
            w3 = np.repeat(w[:, :, np.newaxis], 3, axis=2)
            acc += warped.astype(np.float32) * w3
            weight_sum += w

        weight_sum_3 = np.repeat(np.maximum(weight_sum, 1e-6)[:, :, np.newaxis], 3, axis=2)
        result = (acc / weight_sum_3).astype(np.uint8)
        return result

    def crop_black_edges(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped = image[y:y+h, x:x+w]
        return cropped

    def stitch(self):
        Hs = self.compute_pairwise_homographies()
        transforms = self.accumulate_to_center(Hs)
        for i, t in enumerate(transforms):
            if t is None:
                print(f"Warning: no transform for image {i}; using identity")
                transforms[i] = np.eye(3)

        canvas_w, canvas_h, offset = self.make_canvas(transforms)
        print(f"Canvas size: {canvas_w} x {canvas_h}")

        pano = self.warp_and_blend(transforms, (canvas_w, canvas_h), offset)
        cropped = self.crop_black_edges(pano)
        return cropped

def cylindrical_warp(img, focal_length):
    """
    Warps an image into cylindrical coordinates to reduce perspective distortion.
    """
    h, w = img.shape[:2]
    K = np.array([[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]])
    
    # Coordinate grid
    y_i, x_i = np.indices((h, w))
    
    # Convert to standard coordinates relative to center
    x_c = x_i - w / 2
    y_c = y_i - h / 2
    
    # Cylindrical coordinates conversion
    theta = x_c / focal_length
    h_range = y_c / focal_length
    
    x_sphere = np.sin(theta)
    y_sphere = h_range
    z_sphere = np.cos(theta)
    
    # Project back to flat image plane
    x_flat = focal_length * (x_sphere / z_sphere) + w / 2
    y_flat = focal_length * (y_sphere / z_sphere) + h / 2
    
    # Use remap to warp
    warped = cv2.remap(img, x_flat.astype(np.float32), y_flat.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    # Crop the curved black borders created by the warp itself
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    x, y, w_new, h_new = cv2.boundingRect(thresh)
    
    return warped[y:y+h_new, x:x+w_new]

def clean_edges(img):
    """
    Aggressively cleans edge artifacts using Navier-Stokes inpainting.
    Threshold increased to 30 to catch lighter gray noise.
    """
    # 1. Crop to valid bounding box
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)
    crop = img[y:y+h, x:x+w]
    
    # 2. Identify "dull" edge noise
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # INCREASED THRESHOLD: 15 -> 30
    # This catches "lighter" dark gray pixels that looked like dull noise.
    _, mask = cv2.threshold(gray_crop, 5, 255, cv2.THRESH_BINARY_INV)
    
    # 3. Moderate Dilation
    # INCREASED ITERATIONS: 1 -> 2
    # This ensures we cover the entire "fringe" of the edge.
    kernel = np.ones((3,3), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=2)
    
    # 4. Inpaint with Navier-Stokes (NS)
    # INCREASED RADIUS: 3 -> 5
    # Helps blend the stronger correction smoothly.
    print(f"Cleaning edge artifacts (Aggressive: Thresh=30, Iter=2)...")
    cleaned = cv2.inpaint(crop, mask_dilated, 5, cv2.INPAINT_NS)
    
    return cleaned

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('list_file', help='text file listing image paths (one per line)')
    parser.add_argument('--resize', type=float, default=1.0, help='optional scale factor to resize input images')
    parser.add_argument('--out', type=str, default='stitched_output.jpg', help='output filename')
    args = parser.parse_args()

    paths = resolve_paths(args.list_file)
    print('Resolved paths:')
    for p in paths:
        print('  ', p)

    imgs, failed = load_images(paths, resize=args.resize if args.resize and args.resize>0 else None)
    
    if len(imgs) < 2:
        print('Not enough valid images. Exiting.')
        sys.exit(1)

    # 1. APPLY CYLINDRICAL WARP
    print("Applying Cylindrical Warp to images...")
    # Estimate focal length as the width of the image
    focal_length = imgs[0].shape[1] 
    
    cylindrical_imgs = []
    for img in imgs:
        warped_img = cylindrical_warp(img, focal_length)
        cylindrical_imgs.append(warped_img)
        
    imgs = cylindrical_imgs

    # 2. STITCH
    matcher = Matchers()
    stitcher = Stitcher(imgs, matcher)
    final_pano = stitcher.stitch()

    # --- REPLACE THE PREVIOUS CROP LOGIC WITH THIS ---
    print("Maximizing image area with inpainting...")
    final_output = clean_edges(final_pano)
    
    cv2.imwrite(args.out, final_output)
    print('Saved panorama to', args.out)