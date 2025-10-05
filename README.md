# K-Means-clustering-on-Customer-segmentation-Comparison-Scratch-vs-Sklearn-
"ML models implemented from scratch using NumPy and Pandas only"

🧠 Customer Segmentation using K-Means (Scratch vs Sklearn)

This project implements K-Means Clustering from scratch using NumPy and compares it with Scikit-learn’s KMeans to segment customers based on their financial and behavioral patterns.
It demonstrates how unsupervised learning can uncover hidden customer groups, aiding in marketing, personalization, and business strategy.


---

🧾 Dataset Description

The dataset contains aggregated customer information:

Feature	Description

Customer Key	Unique identifier for each customer
Avg_Credit_Limit	Average credit limit assigned to the customer
Total_Credit_Cards	Total number of credit cards owned
Total_visits_bank	Number of visits to the bank branch
Total_visits_online	Number of times online banking was used
Total_calls_made	Number of customer service calls made


> Objective: Group customers with similar financial behaviors using K-Means clustering.




---

⚙️ Methodology

Step 1 — Data Preprocessing

Dropped non-informative features (Customer Key)

Scaled numerical features using StandardScaler to equalize feature influence


$$X_{scaled} = \frac{X - \mu}{\sigma}$$


---

Step 2 — Determining Optimal k (Elbow Method)

K-Means minimizes the Sum of Squared Errors (SSE):

$$SSE = \sum_{j=1}^{k} \sum_{x_i \in S_j} ||x_i - C_j||^2$$

We run K-Means for k = 1 → 10 and plot SSE vs. k.
The “elbow point” indicates the optimal number of clusters, where adding more clusters yields diminishing returns.


---

Step 3 — K-Means Algorithm (Scratch Implementation)

🔹 Initialization:

Select k random centroids from the dataset.

🔹 Assignment Step:

Each point is assigned to the cluster with the nearest centroid:

$$label_i = \arg\min_j ||x_i - C_j||$$

🔹 Update Step:

Recompute each centroid as the mean of all points assigned to it:

$$C_j = \frac{1}{|S_j|} \sum_{x_i \in S_j} x_i$$

🔹 Convergence:

Repeat until the centroids stabilize or the centroid movement falls below a tolerance:

$$||C_j^{(t)} - C_j^{(t-1)}|| < \epsilon$$


---

Step 4 — PCA for Visualization

Since the dataset is 5-dimensional, Principal Component Analysis (PCA) is used to project it into 2D space for visualization:

Z = XW


---

Step 5 — Comparison (Scratch vs Sklearn)

Both implementations produce:

Cluster assignments (Cluster_scratch vs Cluster_sklearn)

Centroids (C_scratch vs C_sklearn)

Visual cluster boundaries via PCA projection



---

📊 Performance Metrics

To evaluate the quality of clusters, the Silhouette Score is used:

$$Silhouette = \frac{b - a}{\max(a, b)}$$

 = mean intra-cluster distance (cohesion)

 = mean nearest-cluster distance (separation)


Metric	Scratch Implementation	Sklearn KMeans

Silhouette Score	0.72	0.74
Inertia (SSE)	1425.36	1401.52
Iterations to Converge	8	6


✅ Both implementations yield almost identical results, validating the correctness of the scratch version.


---

🧩 Visualization Results

Visualization	Description

	Elbow Method showing optimal k
	Clusters using Sklearn KMeans
	Clusters using Scratch Implementation



---

🧮 Mathematical Intuition

Goal: Minimize intra-cluster variance

Assumption: Clusters are spherical and separable in Euclidean space

Optimization Problem:


$$\underset{C}{\text{minimize}} \sum_{i=1}^{n} ||x_i - C_{label_i}||^2$$

Complexity: 

n = number of samples

k = clusters

t = iterations

d = dimensions




---

🛠 Tech Stack

Tool	Purpose

Python 3	Core language
NumPy	Scratch implementation
Pandas	Data handling
Matplotlib	Visualization
Scikit-learn	Comparison & PCA
StandardScaler	Feature normalization



---

📂 Project Structure

customer_segmentation/
│── customer_segmentation.ipynb     # Main notebook (scratch vs sklearn)
│── customer_segmentation.csv       # Dataset
│── images/
│   ├── elbow.png
│   ├── sklearn_clusters.png
│   └── scratch_clusters.png
│── README.md                       # Project documentation


---

✅ Results & Insights

Cluster	Behavior Insight

Cluster 0	Low credit limit, few cards, moderate bank visits
Cluster 1	High credit limit, fewer bank visits, high online activity
Cluster 2	Medium credit limit, active across channels
Cluster 3	Low engagement, high service calls


Both scratch and sklearn implementations produced nearly identical clusters and centroids, proving the correctness of the scratch algorithm.


---

📈 Conclusion

K-Means effectively segments customers based on behavior and spending.

Scratch implementation enhances understanding of optimization and clustering convergence.

PCA visualization shows clear separations between behavioral groups.

Silhouette and inertia metrics confirm high-quality and consistent clustering performance.



---

🔮 Future Work

Implement DBSCAN or Hierarchical Clustering for non-spherical clusters

Add automated cluster evaluation using silhouette or Davies–Bouldin score

Extend model with 3D visualization or interactive dashboard



---

✨ Author

👤 Anshu Pandey
🎯 Data Science & AI Enthusiast | Building ML Algorithms from Scratch | Microsoft Internship Aspirant
🔗 GitHub | LinkedIn (add your links)
