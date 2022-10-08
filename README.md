# lof
Assigning outliers to detected by the Local Outlier Factor to the nearest neighbor (with iris dataset)
Assigning outliers to detected by the Local Outlier Factor to the nearest neighbor (with iris dataset)

Local Outlier Factor is an unsupervised outlier determination algorithm. Computes the local density deviation of a given data point with respect to its neighbors. Proposed by Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng and Jörg Sander in 2000.

![image](https://user-images.githubusercontent.com/89804884/194721193-48294406-38af-47aa-8332-2c3e041494c9.png)

As can be seen from the diagram above, the points that are far from the clusters are outliers. To determine these, Local Outlier Factor uses k-distance.

![image](https://user-images.githubusercontent.com/89804884/194721238-ade40c7e-e5b2-4a1d-ab20-dc1696b3d294.png)

In this article, I will skip to coding Local Outlier Factor rather that theory and formulas .
For more detailed information, see the resources links.

Here's how Local Outlier Factor is:
First, a dataset is selected. Data is fitting to Local Outlier Factor from within Scikit Learn.

Then the scores are determined with negative_outlier_factor_ attribution. Scores are sorting and observed.
Then the hyperparameter is selected by an outside intervention. As a threshold value. Then this threshold value is assigned to all outliers.

That's all.

It's actually a simple process, but we can ask the following question when we look more closely: Why do we assign same value (threshold value) to all outliers.

Looking at the image below, does it make sense to assign far neighbor's values to the outlier A?

Wouldn't it be better to find the nearest neighbor of an outlier and assign it to it?

That's what we're going to do today.

You can assign the outliers values to the nearest neighbor in the Local Outlier Factor process by following the codes below step by step.

Enjoy!

![image](https://user-images.githubusercontent.com/89804884/194721254-c13858ff-7c11-4953-9e8c-5429292cb643.png)

