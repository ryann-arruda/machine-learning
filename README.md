# Machine Learning

This repository was created based on information collected during my Machine Learning studies. Therefore, it will be constantly updated with new information as my knowledge improves.

## Definition

**Machine Learning** consists of studying and building intelligent agents, which can perceive the environment through sensors and act in the environment through actuators.

Another possible definition for **Machine Learning** is an algorithm that responds efficiently to situations that have not been presented.

## Types of Machine Learning

* **Supervised Learning:** The learning model receives a set of inputs consisting of data and class labels.

    * **Classification:** The set of values is nominal and finite;

        **Example:** Classify whether an instance is spam (A) or not spam (B).

        <figure>
            <img src="https://github.com/ryann-arruda/machine-learning/assets/53544629/22561e82-5570-42a4-a6b9-22110df00299" alt="Classification">
            <br>
            <figcaption><strong>Fonte: </strong>https://www.javatpoint.com/classification-algorithm-in-machine-learning</figcaption>
        </figure>

    * **Regression:** The set of values is ordered and infinite.

        **Example:** Infer the price (Y) of a car based on data features (X).

        <figure>
            <img src="https://github.com/ryann-arruda/machine-learning/assets/53544629/8c2e1a78-ab1e-437d-bfd4-5b9e8fca780b" alt="Regression">
            <br>
            <figcaption><strong>Fonte: </strong>https://www.iguazio.com/glossary/regression/</figcaption>
        </figure>

* **Unsupervised Learning:** The learning model receives a set of inputs made up of data and will attempt to create clusters based on the characteristics of the data.

    **Example:** Recognize images of cars (cluster 1), motorcycles (cluster 2) and planes (cluster 3) by grouping them into groups.

    <figure>
        <img src="https://github.com/ryann-arruda/machine-learning/assets/53544629/a4856d27-bcd0-4a0a-b607-476d919b95fd" alt="Clustering">
        <br>
        <figcaption><strong>Fonte: </strong>https://training.galaxyproject.org/training-material/topics/statistics/tutorials/clustering_machinelearning/tutorial.html</figcaption>
    </figure>

* **Semi-supervised Learning:** The learning model receives a set of input data with and without labels. Then unsupervised learning will be applied to group the data according to characteristics. Finally, supervised learning will be applied in order to expand the labels present in each group to all elements in that group.

    **Example:** A typical example of semi-supervised learning occurs in document classification.

* **Reinforcement Learning:** The model will learn through trial and error, seeking to maximize its performance. In this learning process, there are rewards if the model achieves its objectives and penalties if it takes any action that reduces its performance.

    **Example:** A common scenario where this learning approach is used is in games, for example. In chess, if you lose a piece, you receive a penalty, and if you capture the enemy piece, you receive a reward. So, the more pieces you capture, the more rewards you will receive.

After the brief explanation given above about Machine Learning, the next sections will explain the attributes of the dataset and will cover each learning and some techniques used in each .ipynb file linked to this section.

## Supervised Learning

### Classification

#### Explanation of Dataset Attributes

For supervised classification learning, we will use the '[Breast Cancer Wisconsin](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)' dataset. However, first, we will understand the attributes of this dataset and how they contribute to the problem at hand.

* **id**: instance identifier;

* **diagnosis**:
    1. M = malignant;
    2. B = benign.

* **radius**: average of the distances from the center to the perimeter points. This attribute can be divided into three possible values, namely:
    1. radius_mean: average of the "radius" feature calculated for each image;
    2. radius_se: standard error of the "radius" feature calculated for each image;
    3. radius_worst: mean of the three largest values of the "radius" feature calculated for each image.

* **texture**: standard deviation of gray-scale values. This attribute can be divided into three possible values, namely:
    1. texture_mean: average of the "texture" feature calculated for each image;
    2. texture_se: standard error of the "texture" feature calculated for each image;
    3. texture_worst: mean of the three largest values of the "texture" feature calculated for each image.

    **OBS.:** Gray-scale indicates that different shades of gray are used to represent different densities of breast tissue.

* **perimeter**: indicates the total length of the edge of the cell nucleus. This attribute can be divided into three possible values, namely:
    1. perimeter_mean: average of the "perimeter" feature calculated for each image;
    2. perimeter_se: standard error of the "perimeter" feature calculated for each image;
    3. perimeter_worst: mean of the three largest values of the "perimeter" feature calculated for each image.

* **area**: indicates the measurement of the surface occupied by the cell nucleus. This attribute can be divided into three possible values, namely:
    1. area_mean: average of the "area" feature calculated for each image;
    2. area_se: standard error of the "area" feature calculated for each image;
    3. area_worst: mean of the three largest values of the "area" feature calculated for each image.    

* **smoothness**: it indicates the local variation in the lengths of the rays, that is, it allows us to understand how smooth or irregular the surface of the cell nucleus is. This attribute can be divided into three possible values, namely:
    1. smoothness_mean: average of the "smoothness" feature calculated for each image;
    2. smoothness_se: standard error of the "smoothness" feature calculated for each image;
    3. smoothness_worst: mean of the three largest values of the "smoothness" feature calculated for each image. 

* **compactness**: this attribute is represented by the following mathematical expression:

$$compactness = \frac{perimeter^2}{area - 1.0}$$

It indicates how compacted the cell nucleus is, that is, how regular or irregular the surface of the cell nucleus is. This attribute can be divided into three possible values, namely:

1. smoothness_mean: average of the "smoothness" feature calculated for each image;
2. smoothness_se: standard error of the "smoothness" feature calculated for each image;
3. smoothness_worst: mean of the three largest values of the "smoothness" feature calculated for each image. 

* **concavity**: it indicates the severity of the concave portions of the contour, that is, how accentuated or deep the concave parts of the cell nucleus are. This attribute can be divided into three possible values, namely:

    1. concavity_mean: average of the "concavity" feature calculated for each image;
    2. concavity_se: standard error of the "concavity" feature calculated for each image;
    3. concavity_worst: mean of the three largest values of the "concavity" feature calculated for each image. 

* **concave points**: number of concave portions of the contour. This attribute can be divided into three possible values, namely:

    1. concave points_mean: average of the "concave points" feature calculated for each image;
    2. concave points_se: standard error of the "concave points" feature calculated for each image;
    3. concave points_worst: mean of the three largest values of the "concave points" feature calculated for each image. 

* **symmetry**: it indicates how symmetrical the contour of the cell nucleus is. This attribute can be divided into three possible values, namely:

    1. symmetry_mean: average of the "symmetry" feature calculated for each image;
    2. symmetry_se: standard error of the "symmetry" feature calculated for each image;
    3. symmetry_worst: mean of the three largest values of the "symmetry" feature calculated for each image. 

* **fractal dimension**: it indicates a measure of the complexity of the contour shape of the cell nucleus. Fractal dimension is calculated through a technique called “coastal approximation,” which is an approach to measuring the complexity of a line or contour. The formula for calculating the fractal dimension can be seen below.

$$fractalDimension = coastlineApproximation - 1$$

In general terms, the greater the fractal dimension, the more complex and irregular the cell nucleus.

**OBS.:** It should be borne in mind that the nucleus of a cell has different and uniform shapes.

#### Code

After understanding what each attribute of our dataset means, we will now go through the Pre-Processing, Choosing a Target Attribute, Dimensionality Reduction (if necessary), Supervised Learning and finally Final Considerations sections.

The previously mentioned sections will be included in the Google Colab notebook [machine-learning-classification.ipynb](https://github.com/ryann-arruda/machine-learning/blob/main/machine_learning_classification.ipynb). To see them, enter the notebook.

After understanding how to implement supervised classification learning, we will now continue our studies by addressing the second type of supervised learning: regression.

**NOTE:** It's important to highlight that in the next sections and in the Colab Notebooks, explanations previously made in this section will not be repeated. Therefore, only brief comments will be made on the approaches and decisions taken, while only techniques or methods not previously explained will be detailed.

### Regression

