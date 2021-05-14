# spherical-classification

Code for spherical classification method including the modeling process as well as the categorization techniques. The method is based on the work of Zhengyu Ma1 and  Hong Seo Ryoo (Spherical Classification of Data, a New Rule-Based Learning Method)[Ma-Ryoo2020_Article_SphericalClassificationOfDataA.pdf](https://github.com/mariannakaloutsi/spherical-classification/files/6477397/Ma-Ryoo2020_Article_SphericalClassificationOfDataA.pdf)
 . 

Modeling includes creating categorized clusters based on the training samples which are characterized by a center and two radial borders, a conservative and a liberal one. Code includes the categorization technique proposed by the paper, as well as two alternative ones based only on the distances between the test samples and the centers of the created clusters.

Code is presented both as an function and a class (with a 'fit' and 'predict' method)
