# A4NT - Translating writing styles for anonymity
Code for the paper "A4NT: Author Attribute Anonymity by Adversarial Training of Neural Machine Translation" presented in Usenix 2018. [Link](https://www.usenix.org/conference/usenixsecurity18/presentation/shetty)

# Pretrained models and dataset
Pre-processed datasets used in our paper are available below in json format
* [Political speech dataset](https://datasets.d2.mpi-inf.mpg.de/rakshith/a4nt_usenix/dataset/dataset_speech.json)
* [Blog dataset for age and gender](https://datasets.d2.mpi-inf.mpg.de/rakshith/a4nt_usenix/dataset/dataset_blog.json)

Pre-trained translator models are available below:
* [Age translator](https://datasets.d2.mpi-inf.mpg.de/rakshith/a4nt_usenix/translate_models/age_translator.pth.tar) for adult (23-40 years) to teenager(13-19 years) and vice-versa.
* [Gender translator](https://datasets.d2.mpi-inf.mpg.de/rakshith/a4nt_usenix/translate_models/gender_translator.pth.tar)
* [Speech translator](https://datasets.d2.mpi-inf.mpg.de/rakshith/a4nt_usenix/translate_models/speechObamaAndTrump_translator.pth.tar) for Obama to Trump and vice-versa. 

# Usage instructions
Code is written in python 2.7 and uses pytorch 3.x library.

More instructions coming soon

# Bibtex
If you find this code useful in your work, please cite the paper.
```
@inproceedings{shetty_USENIX2018,
TITLE = {$A^{4}NT$: Author Attribute Anonymity by Adversarial Training of Neural Machine Translation},
AUTHOR = {Shetty, Rakshith and Schiele, Bernt and Fritz, Mario},
LANGUAGE = {eng},
ISBN = {978 -1- 931971- 4 6 -1},
PUBLISHER = {USENIX Association},
YEAR = {2018},
BOOKTITLE = {Proceedings of the 27th USENIX Security Symposium},
PAGES = {1633--1650},
ADDRESS = {Baltimore, MD, USA},
}
```
