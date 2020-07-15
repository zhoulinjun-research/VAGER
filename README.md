# Learning to Learn Image Classifiers with Visual Analogy

By **Linjun Zhou, Peng Cui, Shiqiang Yang, Wenwu Zhu, Qi Tian**.

Tsinghua University & Huawei Noah's Ark Lab.

### Profile
This repository is for our CVPR2019 paper, named **Learning to Learn Image Classifiers with Visual Analogy**, which could be found [here](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhou_Learning_to_Learn_Image_Classifiers_With_Visual_Analogy_CVPR_2019_paper.pdf). If you'd like to use our model in your own research, please cite:

    @inproceedings{zhou2019learning,
      title={Learning to learn image classifiers with visual analogy},
      author={Zhou, Linjun and Cui, Peng and Yang, Shiqiang and Zhu, Wenwu and Tian, Qi},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      pages={11497--11506},
      year={2019}
    }

### Usage
1. Split ImageNet into M base classes and N novel classes, train an AlexNet model on all base classes. (In our paper, M=800 and N=200)
2. Extract 4096-dimension fc7 features from the AlexNet model from step 1, for each base class and novel class.
3. Put all features of base classes in `features/base/`, one .npy file for one class, named `[0-799].npy`, each containing a [K, 4096] matrix, where K is the image number in the corresponding class.
4. Put all features of novel classes in `features/novel/`, one .npy file for one class, named `[0-199].npy`.
5. Put all features of validation images in base classes in `features/val/`, one .npy file for one class, named `[0-799].npy`.
6. `ls src/` and then execute `python preprocessing.py`
7. Execute `python ddim.py`
8. Execute `python data_convert.py`, step 6, 7, 8 is a preliminary feature dimension deduction step.
9. Execute `python training.py`, training base classes with VAGER.
10. Execute `python multi_cls.py`, generalization to novel classes, and then testing for accuracy.
