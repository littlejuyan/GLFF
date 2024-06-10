## [GLFF: Global and Local Feature Fusion for Face Forgery Detection](https://arxiv.org/pdf/2211.08615.pdf)
![Teaser](https://github.com/littlejuyan/GLFF/blob/main/teaser3.png)

### DF^3 Dataset

**Our DF^3 dataset is a large-scale and highly-diverse face forgery dataset by considering 6 state-of-the-art generation models and 5 post-processing operations to approach the real-world applications. Please fill out this form to get the access of our dataset: [Form](https://docs.google.com/forms/d/1STdUMSbrG-f9lWcgSEpZpi13ntg-aznPJqUFGdeTP6w/viewform?edit_requested=true)**



### GLFF Detection Method


## Dataset
Prepare your training and evaluation dataset according to [this repo](https://github.com/peterwang512/CNNDetection) and [this repo](https://github.com/littlejuyan/FusingGlobalandLocal)


## Training
We provide an example script to train our model by running `bash train.sh`

## Testing
We provide an example script to test our model by running `bash test.sh`. 


### Acknowledgments
- This repository borrows partially from [CNNDetection](https://github.com/peterwang512/CNNDetection), [BeyondtheSpectrum](https://github.com/SSAW14/BeyondtheSpectrum), [Nodown](https://github.com/grip-unina/GANimageDetection), [FusingGlobalandLocal](https://github.com/littlejuyan/FusingGlobalandLocal).

- This work is supported by the US Defense Advanced Research Projects Agency (DARPA) Semantic Forensic (SemaFor) program.

### Citation
If you find this useful for your research, please consider citing this bibtex:

@article{ju2023glff,
  title={Glff: Global and local feature fusion for ai-synthesized image detection},
  author={Ju, Yan and Jia, Shan and Cai, Jialing and Guan, Haiying and Lyu, Siwei},
  journal={IEEE Transactions on Multimedia},
  year={2023},
  publisher={IEEE}
}

