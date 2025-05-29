# Multi-scale_FE (Implementation of "A Fusion-based Enhancing Method for Weakly Illuminated Images")
Python implementation of the algorithm in the paper "[A Fusion-based Enhancing Method for Weakly Illuminated Images](https://www.sciencedirect.com/science/article/pii/S0165168416300949)".

---

## Usage

**a. Go to the project directory.**

```shell
cd Multi-scale_FE/
```

**b. Generate illumination enhanced image.**

```shell
python multi_scale_fe.py --input_path ${INPUT_PATH} (optional: --level ${LEVEL})
```

Please replace `${INPUT_PATH}` and `${LEVEL}` in the shell of the command line with the real input image path and the number of layers.

---

## Optimizations

The `multi_scale_fe.py` script has undergone several optimizations to improve its performance and efficiency:

*   **`box_filter` Optimization:** The original cumulative sum-based `box_filter` was replaced with `cv2.boxFilter(..., normalize=False, borderType=cv2.BORDER_REPLICATE)`, leveraging OpenCV's highly optimized implementation for sum-based box filtering.
*   **`guided_filter` Efficiency:** The calculation of the window area (`N`) within the `guided_filter` was changed from a `box_filter` call on an array of ones to a direct scalar computation (`N = (2 * rad + 1)**2`), saving a full image filtering operation.
*   **OpenCV Pyramid Functions:** The manual Gaussian pyramid construction functions (`pyramid_reduce` and `pyramid_expand`) were replaced with OpenCV's optimized `cv2.pyrDown` and `cv2.pyrUp` functions, respectively. The `generate_pyramid` function was updated to use these standard functions, ensuring correct size handling for Laplacian pyramid construction.
*   **Vectorized Structuring Element:** The loop-based generation of the structuring element (`struct_elem`) in `fusion_based_method` was replaced with direct NumPy array assignments for improved efficiency.
*   **Redundancy Reduction:** Unnecessary data type conversions (e.g., redundant `np.double()` calls on already float64 arrays) were removed in several parts of the code, particularly in pyramid generation and image resizing loops.
*   **General Python Cleanups:**
    *   Unused imports (`math`, `scipy.signal`) were removed.
    *   `math.pi` was replaced with `np.pi`.
    *   Non-essential `matplotlib.pyplot.show()` calls were commented out to make the script suitable for non-interactive execution, relying on `plt.imsave()` for output.

These optimizations contribute to a more streamlined, faster, and more efficient execution of the image enhancement process.
