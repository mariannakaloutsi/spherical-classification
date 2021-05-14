# spherical-classification

Code for spherical classification method including the modeling process as well as the categorization techniques. The method is based on the work of Zhengyu Ma and  Hong Seo Ryoo [Ma-Ryoo2020_Article_SphericalClassificationOfDataA.pdf](https://github.com/mariannakaloutsi/spherical-classification/files/6477397/Ma-Ryoo2020_Article_SphericalClassificationOfDataA.pdf). 

Modeling includes creating categorized clusters based on the training samples which are characterized by a center and two radial borders, a conservative and a liberal one. Code includes the categorization technique proposed by the paper, as well as two alternative ones based only on the distances between the test samples and the centers of the created clusters.

Code is presented both as an function and a class (with a 'fit' and 'predict' method)
The function needs as input:
      the directory of the dataset file 
      the column that contains the endpoint
      the training/test ratio
      True/False regarding feature selection
      number of final features in case of feature selection
      feature selection function (chi2, f_classif etc)
      True/False regarding printing results
      Name of solver used for the Linear Programming problem
      list containing the categorization techniques used (1 refers to the one proposed by the original paper while 2 and 3 refer to two alternative ones proposed by [paper name here])

NOTES:
1.	File is a .csv file, properly edited so that it is in table form. The endpoint column in included in the dataset.
2.	The table has columns that correspond to samples features and the target points as well as rows that correspond to the different entries (samples). The table must not contain empty cells, characters, infs or NaNs
3.	The endpoint column is in binary form: 1 for non-toxic and 2 for toxic samples

     
The class needs as input:
      the directory of the training dataset file in .csv form
      the directory of the test dataset file in .csv form
      the column that contains the endpoint
      True/False regarding feature selection
      number of final features in case of feature selection
      feature selection function (chi2, f_classif etc)
      Name of solver used for the Linear Programming problem
      list containing the categorization techniques used 

NOTES:
1. A class is created with inputs the training and test. In the test set, the endpoints column is  missing. The dataset is split beforehand into a training and a test set.
2.	After an instance is created, the model is fitted to the training set with the fit() method. Then, new predictions are made for the test set by using the predict(clusters) method.

