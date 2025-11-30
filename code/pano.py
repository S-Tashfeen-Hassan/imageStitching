import cv2
import numpy as np
import time
import sys
from matchers import Matchers


class Stitch:
    def __init__(self, list_file):
        self.list_file = list_file

        with open(list_file, "r") as f:
            filepaths = [line.strip() for line in f.readlines()]

        print("Loaded filenames:", filepaths)

        # Load & resize images
        self.images = [
            cv2.resize(cv2.imread(path), (480, 320))
            for path in filepaths
        ]

        self.count = len(self.images)
        self.left_list = []
        self.right_list = []
        self.center_image = None

        self.matcher_obj = Matchers()
        self.prepare_lists()

    # ------------------------------------------------------

    def prepare_lists(self):
        print(f"Number of images: {self.count}")

        self.center_idx = self.count // 2
        print(f"Center index: {self.center_idx}")

        self.center_image = self.images[self.center_idx]

        for i, img in enumerate(self.images):
            if i <= self.center_idx:
                self.left_list.append(img)
            else:
                self.right_list.append(img)

        print("Image groups prepared.")

    # ------------------------------------------------------

    def left_shift(self):
        base = self.left_list[0]

        for next_img in self.left_list[1:]:
            H = self.matcher_obj.match(base, next_img, 'left')
            print("Homography:", H)

            if H is None:
                continue

            H_inv = np.linalg.inv(H)
            print("Inverse H:", H_inv)

            # Original corner calculation
            ds = np.dot(H_inv, np.array([base.shape[1], base.shape[0], 1]))
            ds = ds / ds[-1]

            f1 = np.dot(H_inv, np.array([0, 0, 1]))
            f1 = f1 / f1[-1]

            offset_x = abs(int(f1[0]))
            offset_y = abs(int(f1[1]))

            # FIX: make a bigger canvas to avoid broadcasting error
            canvas_width = base.shape[1] * 2
            canvas_height = base.shape[0] * 2

            H_inv[0, 2] += offset_x
            H_inv[1, 2] += offset_y

            tmp = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            # Warp base image into canvas
            warped_base = cv2.warpPerspective(base, H_inv, (canvas_width, canvas_height))
            tmp[:warped_base.shape[0], :warped_base.shape[1]] = warped_base

            # Paste next image
            tmp[offset_y:offset_y + next_img.shape[0],
                offset_x:offset_x + next_img.shape[1]] = next_img

            base = tmp

        self.left_image = base

    # ------------------------------------------------------

    def right_shift(self):
        for next_img in self.right_list:
            H = self.matcher_obj.match(self.left_image, next_img, 'right')
            print("Homography:", H)

            if H is None:
                continue

            h, w = next_img.shape[:2]

            txyz = np.dot(H, np.array([w, h, 1]))
            txyz = txyz / txyz[-1]

            canvas_width = self.left_image.shape[1] * 2
            canvas_height = self.left_image.shape[0] * 2

            tmp = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            warped_img = cv2.warpPerspective(next_img, H, (canvas_width, canvas_height))

            # Blend with current left_image
            tmp[:self.left_image.shape[0], :self.left_image.shape[1]] = self.left_image
            tmp = self.mix_and_match(self.left_image, warped_img)

            self.left_image = tmp

    # ------------------------------------------------------

    def mix_and_match(self, left, warped, px=0, py=0):

        h1, w1 = left.shape[:2]
        h2, w2 = warped.shape[:2]

        for y in range(h1):
            for x in range(w1):

                if y + py >= h2 or x + px >= w2:
                    continue

                px_left = left[y, x]
                px_warp = warped[y + py, x + px]

                if np.array_equal(px_warp, [0, 0, 0]):
                    warped[y + py, x + px] = px_left

                # keep left pixel if both are valid
                elif not np.array_equal(px_left, [0, 0, 0]):
                    warped[y + py, x + px] = px_left

        return warped

    # ------------------------------------------------------

    def show(self, name="result"):
        cv2.imshow(name, self.left_image)
        cv2.waitKey(0)

    # ------------------------------------------------------

if __name__ == "__main__":
    try:
        list_file = sys.argv[1]
    except IndexError:
        list_file = "txtlists/files1.txt"

    print("Using list:", list_file)

    stitcher = Stitch(list_file)
    stitcher.left_shift()
    stitcher.right_shift()

    cv2.imwrite("stitched_output.jpg", stitcher.left_image)
    print("Saved output image: stitched_output.jpg")

    cv2.destroyAllWindows()