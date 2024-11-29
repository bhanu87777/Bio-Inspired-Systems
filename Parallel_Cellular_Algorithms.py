import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

def pca_noise_reduction(image, connectivity=8):
    padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    rows, cols = image.shape
    reduced_image = np.zeros_like(image)

    def process_row(row_idx):
        reduced_row = np.zeros(cols)
        for col_idx in range(cols):
            neighbors = []
            if connectivity == 4:
                neighbors = [
                    padded_image[row_idx, col_idx + 1],
                    padded_image[row_idx + 1, col_idx],
                    padded_image[row_idx + 1, col_idx + 2],
                    padded_image[row_idx + 2, col_idx + 1]
                ]
            elif connectivity == 8:
                neighbors = [
                    padded_image[row_idx, col_idx],
                    padded_image[row_idx, col_idx + 1],
                    padded_image[row_idx, col_idx + 2],
                    padded_image[row_idx + 1, col_idx],
                    padded_image[row_idx + 1, col_idx + 2],
                    padded_image[row_idx + 2, col_idx],
                    padded_image[row_idx + 2, col_idx + 1],
                    padded_image[row_idx + 2, col_idx + 2]
                ]
            reduced_row[col_idx] = np.median(neighbors)
        return reduced_row

    with ThreadPoolExecutor() as executor:
        reduced_rows = list(executor.map(process_row, range(rows)))

    for i, reduced_row in enumerate(reduced_rows):
        reduced_image[i] = reduced_row

    return reduced_image

image_path = "Untitled.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
noise_reduced_image = pca_noise_reduction(image, connectivity=8)

cv2.imshow("Original Image", image)
cv2.imshow("Noise Reduced Image", noise_reduced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
