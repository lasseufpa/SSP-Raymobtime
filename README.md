# SSP-Raymobtime
Code and Data for the paper:
Klautau, A., Oliveira, A., Pamplona, I. & Alves, W.. Generating MIMO Channels For 6G Virtual Worlds Using Ray-Tracing Simulations. IEEE Statistical Signal Processing Workshop, 2021.

Download the beam selection dataset from: https://nextcloud.lasseufpa.org/s/mrzEiQXE83YE3kg

Download the channel estimation dataset from: https://nextcloud.lasseufpa.org/s/JdCJSYSWa3rKKAQ

### Python dependencies
If you want to use the already available preprocessed data that we make available, to train and test this baseline
model the only dependencies are:  
* [TensorFlow](https://www.tensorflow.org/install)
* [Scikit-learn](https://scikit-learn.org/stable/install.html)
* [Numpy](https://numpy.org/install/)
* [Matplotlib](https://matplotlib.org/users/installing.html) For plotting

You may install these packages using pip or similar software. For example, with pip:

pip install tensorflow

### Training and validation
After downloading the data, save it to a folder called `SSP_data/bs_baseline_data/` or `SSP_data/ce_baseline_data/` for the beam selection or channel estimation, respectively. Then you can execute the training and test stages. For instance, assuming beam selection, run the following command in the `beam_selection` directory:

```bash
python beam_selection.py
```

* Parameters   
  * (**Optional**) *--plots* plot the accuracy and validation accuracy of your model.


To train a mimo_fixed channel, for instance, use the following command inside `channel_estimation` folder:
```bash
python train.py mimo_fixed
```

* Parameters 
  * (**Required**) *--model_name* for channel estimation simulation, it represents model name and channel data that will be used to train or test.
  * (**Optional**) *--plots* plot the accuracy and validation accuracy of your model.

Also, you should create `models` and `results` folders inside `channel_estimation` directory.

To test a channel estimation model, use `test.py` file:
```bash
python test.py mimo_fixed
```

### Citation

```bibtex
@inproceedings{
    klautau2021,
    title={Generating {MIMO} Channels for {6G} Virtual Worlds Using Ray-tracing Simulations},
    author={Aldebaro Klautau and Ailton De Oliveira and Isabela Trindade and Wesin Alves},
    booktitle={IEEE Statistical Signal Processing Workshop},
    year={2021},
    month={Jul}
}
```
