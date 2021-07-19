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

