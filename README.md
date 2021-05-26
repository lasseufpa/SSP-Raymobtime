# SSP-Raymobtime
Codes and Data for the paper:
Klautau, A., Oliveira, A., Pamplona, I. & Alves, W.. Generating MIMO Channels For 6G Virtual Worlds Using Ray-Tracing Simulations. IEEE Statistical Signal Processing Workshop, 2021.

Beam selection dataset in: https://nextcloud.lasseufpa.org/s/mrzEiQXE83YE3kg
Channel estimation dataset in: https://nextcloud.lasseufpa.org/s/JdCJSYSWa3rKKAQ

### Python dependencies
If you want to use the already available preprocessed data, to train and test this baseline
model the only dependencies are:  
* [TensorFlow](https://www.tensorflow.org/install)
* [Scikit-learn](https://scikit-learn.org/stable/install.html)
* [Numpy](https://numpy.org/install/)
* [Matplotlib](https://matplotlib.org/users/installing.html) For plotting

You may install these packages using pip or similar software. For example, with pip:

pip install tensorflow

### Preprocessing
Before training your model, preprocess the data using:

```bash
python preprocessing.py raymobtime_root data_folder
```
* Parameters
  
  * (**Mandatory**) *raymobtime_root* is the directory where you placed the files related to the Raymobtime dataset, downloaded using one of the scripts available [here](https://github.com/lasseufpa/ITU-Challenge-ML5G-PHY/tree/master/Beam_selection/data).

  * (**Mandatory**) *data_folder* is the directory where you want to place the processed files.

> If *data_folder* doesn't exist, it will be created for you

### Training and validation
After download the data and save at `SSP_data/bs_baseline_data/` or `SSP_data/ce_baseline_data/`, run the following command in the `beam_selection` directory  for beam selection simulation:

```bash
python beam_selection.py
```

* Parameters   
  * (**Optional**) *--plots* plot the accuracy and validation accuracy of your model.


To train a mimo_fixed channel, as instance, use the following command:
```bash
python train.py mimo_fixed
```

* Parameters 
  * (**Required**) *--model_name* for channel estimation simulation, it represents model name and channel data that will be used to train or test.
  * (**Optional**) *--plots* plot the accuracy and validation accuracy of your model.

Also, you should create `models` and `results` folders inside `channel_estimation` directory.

### Citation

```bibtex
@inproceedings{
    klautau2021,
    title={Generating {MIMO} Channels for {6G} Virtual Worlds Using Ray-tracing Simulations},
    author={Klautau, Aldebaro and De Oliveira, Ailton and Trindade, Isabela and Alves, Wesin},
    booktitle={IEEE Statistical Signal Processing Workshop},
    year={2021},
    month={Jul},
        
}
```
