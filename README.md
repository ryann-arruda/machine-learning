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

## Explanation of Dataset Attributes

* id: instance identifier;
* diagnosis:
    1. M = malignant;
    2. B = benign.
* radius: average of the distances from the center to the perimeter points. This attribute can be divided into three possible values, namely:
    1. radius_mean: average of the "radius" feature calculated for each image;
    2. radius_se: standard error of the "radius" feature calculated for each image;
    3. radius_worst: mean of the three largest values of the "radius" feature calculated for each image.
* texture: standard deviation of gray-scale values. This attribute can be divided into three possible values, namely:
    1. texture_mean: average of the "texture" feature calculated for each image;
    2. texture_se: standard error of the "texture" feature calculated for each image;
    3. texture_worst: mean of the three largest values of the "texture" feature calculated for each image.

    **OBS.:** Gray-scale indicates that different shades of gray are used to represent different densities of breast tissue.
* perimeter:
    1. perimeter_mean: average of the "perimeter" feature calculated for each image;
    2. perimeter_se: standard error of the "perimeter" feature calculated for each image;
    3. perimeter_worst: mean of the three largest values of the "perimeter" feature calculated for each image.

**OBS.:** It should be borne in mind that the nucleus of a cell has different and uniform shapes.

## Supervised Learning

### Classification

Before you start programming, you need to understand what the Dataset is about and what information it contains in each column, as well as how much they contribute to the problem.