# Motion Learning Toolbox

The Motion Learning Toolbox is a Python library designed to facilitate the preprocessing of motion tracking data in extended reality (XR) setups. It's particularly useful for researchers and engineers wanting to use XR tracking data as input for machine learning models. Originally developed for academic research targeting the identification of XR users by their motions, this toolbox includes a variety of data encoding methods that enhance machine learning model performance.

The library is still in active development and we continue to add and update functionality. Therefore, any feedback and contributions are very welcome!

## Origins

The methods and techniques in this library have been developed, analysed and applied in our papers:

1. ["Comparison of Data Encodings and Machine Learning Architectures for User Identification on Arbitrary Motion Sequences"](https://ieeexplore.ieee.org/document/10024474), 2022, C. Rack, A. Hotho and M. E. Latoschik, IEEE AIVR
2. ["Who Is Alyx? A new Behavioral Biometric Dataset for User Identification in XR"](https://arxiv.org/abs/2308.03788), 2023, C. Rack, T.  Fernando, M. Yalcin, A. Hotho, and M. E. Latoschik, *arXiv e-prints*
3. ["Extensible Motion-based Identification of XR Users using Non-Specific Motion Data"](https://arxiv.org/abs/2302.07517), 2023, C. Rack, K. Kobs, T. Fernando, A. Hotho, M. E. Latoschik, *arXiv e-prints*

## Importance of Data Encoding
The core features of this library target the encoding of tracking data. Identifying users based on their motions usually starts with a raw stream of positional and rotational data, which we term scene-relative (SR) data. While SR data is informative, it includes information that can distort the learning objectives of identification models.  For instance, SR data includes not just user-specific characteristics but also information about the user's arbitrary position in the VR scene—features that don't contribute to user identity. To alleviate this, Motion Learning Toolbox offers additional encodings, such as:

- **Body-Relative (BR) Data**: Transforms the coordinate system to a frame of reference attached to the user's head, thereby filtering out some of the scene-specific noise.
  
- **Body-Relative Velocity (BRV) Data**: Computes the derivative of BR data over time, isolating the velocity component to focus on actual user movement.

- **Body-Relative Acceleration (BRA) Data**: A further derivative to capture the acceleration features for potentially improved model training.

These alternative encodings are instrumental in enhancing the ability of machine learning models to learn user-specific characteristics by minimizing the amount of irrelevant information. Besides providing these data encoding methods, the Motion Learning Toolbox also provides methods to resample and clean up XR tracking data.

## Setup Instructions
To get started with Motion Learning Toolbox, follow these steps:

1. Install library:
    ```bash
    pip install motion-learning-toolbox
    ```

2. Import the library into your Python script:
    ```python
    import motion_learning_toolbox as mlt
    ```

3. Use like this:
    ```python
    mlt.to_body_relative(...)
    ```

## Features

The following methods are explained in detail and demonstrated in [`examples/demo.ipynb`](examples/demo.ipynb).

- Data Cleanup
    - `fix_controller_mapping` - during calibration, XR systems might assign left and right controllers the wrong way around; this methods checks this and renames the columns if necessary.
    - `resample` – resamples the recording to a constant frame rate, using linear interpolation for positions and Slerping for quaternions.
    - `canonicalize_quaternions` - provides a unique representation for each quaternion, which is desirable for machine learning models.
- Data Encoding
    - `to_body_relative` - encodes scene-relative (SR) data to body-relative (BR) data.
    - `to_velocity` - encodes SR to scene-relative velocity (SRV) data, or BR to body-relative velocity (BRV) data.
    - `to_acceleration` - encodes SR to scene-relative acceleration (SRA) data, or BR to body-relative acceleration (BRA) data.

## Data Format

This library expects input tracking data as Pandas DataFrame. Positional columns should follow the pattern `<joint>_pos_<x/y/z>`. Rotations have to be encoded as quaternions and follow the pattern `<joint>_rot_<x/y/z/w>`. The order of the columns doesn't matter.

In [`examples/data.csv`](examples/data.csv) you find an example CSV file that yields a compatible DataFrame if loaded with `pd.read_csv(examples/data.csv)`.

## Usage Examples

In the [`examples`](examples) folder of the repository, you'll find a Jupyter Notebook named [`demo.ipynb`](examples/demo.ipynb) that demonstrates how to use most of the functions provided by this library. The notebook serves as a practical guide and showcases the functionality with real data.

## Contact

We welcome any discussion, ideas and feedback around this library. Feel free to either open an issue on GitHub or directly contact Christian Rack or Lukas Schach.

## Development

### Build and publish library

1. Bump versions in setup.py and pyproject.toml
2. Build with `python setup.py build` and `python setup.py sdist`
3. Upload to pypi with `twine upload -r pypi dist/* --skip-existing`
   - username: __token__
   - password: <api-token: pypi-...> 
## License Information

<p xmlns:cc="http://creativecommons.org/ns#">
  This work by <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://hci.uni-wuerzburg.de">Christian Rack, Lukas Schach and Marc E. Latoschik</a> is
  licensed under <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0
  <img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1">
  <img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1">
  <img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1">
  <img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1"></a>
</p>