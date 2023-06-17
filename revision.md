
# Machine Learning
- [K Nearest Neighbors](#k-nearest-neighbors-supervised-classification-and-regression)
- [K Means](#k-means-unsupervised-clustering)
- [Binary Linear Classification](#binary-linear-classification)
- [Logistic Regression](#logistic-regression-supervised-linear-classification)
- [Max Margin Classifier (Linear SVM)](#max-margin-classifier--linear-svm-supervised-linear-classification)
- [Boosting (AdaBoost)](#boosting--adaboost-ensemble-method-non-linear-supervised-classification--regression)
- [Polynomial and RBF SVM](#polynomial-and-rbf-svm-non-linear-supervised-classification)
- [Decision Trees and Forests](#decision-trees-and-forests)

| Algorithm | Type | Use cases |
|---|---|---|
| $K$ Nearest Neighbors | Supervised | Classfication / Regression |
| $K$ Means | Unsupervised | Clustering |
| Linear Regression | Supervised | Regression |
| Logistic Regression | Supervised | Classification |
| Perceptron | Supervised | Classification |
| Linear SVM | Supervised | Classification |
| Poly / RBF SVM | Supervised | Classification |
| Polynomial Approximation | Supervised | Regression |
|}AdaBoost | Supervised | Classification |
| Decision Trees | Supervised | Classification / Regression |

- Primary usages are indicated but a lot of algorithms can be adapted to do other tasks (e.g. regression using classification algorithm)

# K Nearest Neighbors (Supervised Classification and Regression)
- Find the $k$ nearest neighbors in the training set and classify the point according to the majority of labels of this nearest neighbors
- Model with $k=1$ has the highest complexity because it can cause a lot of islands in decision boundary → Increasing $k$ results in smoother boundary
- K-NN algorithm is prone to misclassifying points near the decision boundary 

## Unbalanced class

- $k$-NN relies on the _most frequent_ label amongst the neighbors of a point. When there are more training examples of one kind than the other, the better represented class is unduly favored
    - Weighing neighbors by the inverse of their class size
    - Under-sampling the dominant class
    - Augmenting the other class by generating synthetic examples

## Data Reduction - Condensed NN
- Classify a new feature vector $x$ → compute the distance between $x$ and all points in the learning set and find the $k$-th nearest → very slow for a large dataset
- Replace all points in the training set by a smaller set of prototypes (not possible to use centers of gravity, think of a circle inside another)
- Algorithm with $X$ the set of training samples and $P$ the set of prototypes
    1. Initialize $P$ by selecting randomly one sample from $X$
    2. Repeat until there is a prototype of each class
        1. Look for $x \in X$ such that its nearest prototype in $P$ has a different label than itself
        2. Remove $x$ from $X$ and add it to $P$
- Classifying a point only requires comparing it to the $C$ prototypes where $C$ is the number of labels → faster

&nbsp;
&nbsp;
--- 
&nbsp;
&nbsp;

# K-Means (Unsupervised Clustering)
- The training set is not annotated and the system must also learn the classes 
- Group the samples into $K$ clusters where $K$ is a **hyperparameter** assumed to be known or given
- **Center of gravity** $\mu_k$ of cluster $k$ is  $\mu_k = \frac1{N_k} \sum_{i\in C_k} \mathbf{x}_i$ where $N_k = \left\lvert C_k \right\rvert$

## Formalization
- Distances between the points within a cluster should be small and the distances across clusters should be large
- This can be encoded via the distance to cluster centers $\{\mu_1, \dots, \mu_K\}$ :

$$\min \sum_{k=1}^K \sum_{j=1}^{n^k}(x_{i_j^k} - \mu_k)^2$$

- $\{x_{i_1^k}, \dots, x_{i_{n^k}^k}\}$ are the $N_k$ samples that belong to cluster $k$ → difficult minimization problem
- We don’t know what points belong to what cluster
- We don’t know the center of gravity of the clusters

## Lloyd's Algorithm
- Iterative algorithm until convergence (or difference below threshold) or reach a defined maximum number of iterations (too low → bad results)
- **Always converge** but sometimes not to the best (desired) solution → depends on initial conditions
1. Initialize the cluster's center $\{\mu_1, \dots, \mu_K\}$ randomly
2. Until convergence
    1. Assign each point $x_i$ to the nearest cluster (using $\mu_k$)
    2. Update each center $\mu_k$ given the points assigned to it (recompute)

## Hyperparameter $K$
- Elbow method → average within-cluster sum of squares typically decreases towards zero but using too many clusters make the results meaningless
- Elbow of the curve is where the drop in within-cluster distances becomes less significant → choose the point where the changes between values of $K$ become small

## Non-compact data
- K-means clustering exploits the notion of compactness of clusters, sometimes the data is not compact (e.g. two concentric circles)
- Clusters arise from connectivity of points
    - Build connectivity graph with a node for each sample and $W_{ij}=\exp(\frac{-||x_i-x_j||^2}{\sigma^2}) \forall i, j$ (or restrict to $k$ nearest neighbors)
    - Cut the graph to obtain $K$ clusters by **minimizing** normalized cut $Ncut(A, B)=\frac{cut(A,B)}{vol(A)}+\frac{cut(A,B)}{vol(B)}$ where 
        - $cut(A, B)=\sum_{i\in A, j\in B}W_{ij}$
        - $vol(A)=\sum_{i \in A}d_i$
        - $d_i=\sum_jW_{ij}$
    - Normalized cut avoid imbalanced partitions (e.g. single point in $A$ all others in $B$)
- Minimizing the normalized cut can be approximated as a generalized eigenvalue problem (relaxation) : $(D-W)y=\lambda D y$ where 
    - $W \in \mathbb{R}^{N \times N}$ is the similarity matrix between all pairs of points
    - $D \in \mathbb{R}^{N \times N}$ is the diagonal degree matrix, with $D_{ii}=d_i$
    - $(D-W)$ is referred to as the graph Laplacian
    - Size of $y$ is the number of samples
- Solution is then obtained by the eigenvector with the second smallest eigenvalue :
    - Ideally, a positive value in this vector indicates that the corresponding point belongs to one partition, and a negative value to the other
    - Because of the relaxation, this is not so ideal; one then needs to threshold the values (e.g. by taking the median values as threshold for balanced data)
- To obtain more than 2 clusters ($K$- way partition), one can either : 
    1. Recursively apply the 2-way partitioning algorithm → not efficient and unstable
    2. Use multiple (i.e. $K$) eigenvectors
        - Each point is represented as a $K$-dimensional vector
        - Apply $K$-means clustering to the resulting $N$ vectors
        - Interpretation : Dimensionality reduction followed by $K$-means

&nbsp;
&nbsp;
--- 
&nbsp;
&nbsp;

# Binary Linear Classification

- Given an hyperplane $\mathbf{x} \in \mathbb{R}^N, \tilde{\mathbf{w}} \cdot \tilde{\mathbf{x}} = 0$ with $\tilde{\mathbf{x}} = \begin{bmatrix} 1 | \mathbf{x}\end{bmatrix}$ and $\sum_{i=1}^N w_i^2 = 1$.
- Label $y \in \{-1, 1\}$ or $y \in \{0, 1\}$
- The signed distance $h = \tilde{\mathbf{w}} \cdot \tilde{\mathbf{x}}$
- Find $\tilde{\mathbf{w}}$ such that
    - For all or most positive samples : $\tilde{\mathbf{w}} \cdot \tilde{\mathbf{x}} > 0$ (blue)
    - For all or most negative samples : $\tilde{\mathbf{w}} \cdot \tilde{\mathbf{x}} < 0$ (red)

## Perceptron

$$
\min_{\tilde{\mathbf{w}}}  E(\tilde{\mathbf{w}}) =-\sum_{n=1}^N \text{sign}(\tilde{\mathbf{w}} \cdot \tilde{\mathbf{x}})t_n
$$
- Training
    - Set $\tilde{\mathbf{w}_1}$ to $0$
    - Iteratively, pick a random index $n$
        - If $\tilde{x_n}$ is correctly classified, do nothing
        - Otherwise, $\tilde{\mathbf{w}_{t+1}}=\tilde{\mathbf{w}_t}+t_n \times \tilde{\mathbf{x}_n}$
        - Normalize $\tilde{\mathbf{w}_{t+1}}$
- Test : 
$
y(\mathbf{x}, \tilde{\mathbf{w}})=\begin{cases}1 \quad \tilde{\mathbf{w}} \cdot \tilde{\mathbf{x}} \geq 0 \\ -1 \quad \text{otherwise}\end{cases}
$

## Centered Perceptron
- Help reduce bias 
- Increase the convergence speed 
- Center the $\mathbf{x}_n$ so that $w_0 = 0$ ($\mathbf{x}' = \mathbf{x} - \bar{\mathbf{x}}$)

## Convergence Theorem

- $\exist \gamma > 0$ (**margin**) (i.e. data is linearly separable) and a parameter vector $\mathbf{w}^*$ with $||\mathbf{w}^*||=1$, such that $\forall n, t_n(\mathbf{w}^* \cdot \mathbf{x}_n) > \gamma $, perceptron algorithm makes at most $\frac{R^2}{\gamma^2}$ errors, where $R = \max ||\mathbf{x}_n||$
- **Randomizing** indices of samples (order or exploration) at each iteration of the perceptron algorithm helps to achieve better results (same order produce small $\gamma$)
- If there are outliers in data (misclassified points in ground truth), the perceptron still works up to a point but there is no guarantee
- Perceptron has no way to favor one solution over another using $\text{sign}$ function, no difference between close or far from decision boundary (does not maximize margin)

&nbsp;
&nbsp;
--- 
&nbsp;
&nbsp;

# Logistic Regression (Supervised Linear Classification)
- Prediction becomes $y(\mathbf{x}, \tilde{\mathbf{w}}) = \sigma(\tilde{\mathbf{w}} \cdot \tilde{\mathbf{x}})$ where $\sigma$ is the **sigmoid** function (smoother)
$$
\min_{\tilde{\mathbf{w}}} E(\tilde{\mathbf{w}}) = - \sum_{n=1}^N t_n \times ln(y_n) + (1-t_n) \times ln(1 - y_n)
$$


## Sigmoid Function
$$\sigma(a) = \frac{1}{1 + exp(-a)} \quad \quad \frac{\delta \sigma}{\delta a} = \sigma(1 - \sigma)$$
$$\lim_{x \to -\infin}=0 \quad \quad \lim_{x \to \infin}=1$$

- Easy to compute derivatives and infinitely differentiable
- Smoothed step function

## Cross-Entropy
- Convex function → easy to find global minimum

$$E(\tilde{\mathbf{w}}) = - \sum_n t_n \times ln(y_n) + (1-t_n) \times ln(1 - y_n) \\
\nabla E(\tilde{\mathbf{w}}) = \sum_n (y_n - t_n) \tilde{\mathbf{x}}_n \quad \quad 
y_n = \sigma(\tilde{\mathbf{w}} \cdot \tilde{\mathbf{x}}_n)$$

## Matrix form problem
- Training samples can be represented in a matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$ with $D-1$ features and the bias
- The labels are contained in a vector $\mathbf{t} \in \mathbb{R}^{N}$, where $t_i \in \{0, 1\}$.

$$x_i^T \mathbf{w} \quad \implies \quad \left(1\times D\right) \cdot \left(D\times1\right) = 1\times 1 \implies t_i$$

$$\mathbf{X} \cdot \mathbf{w} \quad \implies \quad \left(N\times D\right) \cdot \left(D\times1\right)=N\times 1 \quad \implies \quad \mathbf{t}$$

## Gradient Descent
- Weights initialized randomly (according to a Gaussian distribution)

$$\nabla E(\mathbf{w}) = \sum_i (y_i - t_i)\mathbf{x}_i $$

$$\nabla E(\mathbf{w})= \mathbf{X}^\top\left(\mathbf{y} - \mathbf{t}\right) = \mathbf{X}^\top\left(\sigma(\mathbf{X}\cdot\mathbf{w}) - \mathbf{t}\right) \quad \text{(same shape as } \mathbf{w}\text{)}$$

$$ \mathbf{w}_{[t + 1]}  = \mathbf{w}_{[t]} - \eta\, \nabla E\left(\mathbf{w}_{[t]}\right) $$

## Probabilistic Interpretation
$$y(\mathbf{x}, \mathbf{w}) = P(y=1|\mathbf{x}, \mathbf{w}) =  \sigma(\tilde{\mathbf{w}} \cdot \tilde{\mathbf{x}}) = \frac{1}{1 + \exp(-\tilde{\mathbf{w}} \cdot \tilde{\mathbf{x}})} \in [0, 1]$$

- $y(\mathbf{x}, \tilde{\mathbf{w}}) = 0.5$ if $\tilde{\mathbf{w}} \cdot \tilde{\mathbf{x}}=0$ which means that $\mathbf{x}$ is **on** the decision boundary
- $y(\mathbf{x}, \tilde{\mathbf{w}})$ can be interpreted as the probability that $\mathbf{x}$ belongs to one class or the other
- Finds the **maximum likelihood solution** under the assumption that the noise is Gaussian
- Can differentiate solution and choose the one that separates best the data (because smoother loss function not using only sign)

&nbsp;
&nbsp;
--- 
&nbsp;
&nbsp;

# Max Margin Classifier : Linear SVM (Supervised Linear Classification)

- Find hyperplane that maximizes the margin between the two classes
- Maximizing the margin → less sensitive to outliers in the data
- Data is not linearly separable → max margin classifiers can be very sensitive to noise in the data

## Margin
- Orthogonal distance between the decision boundary and the nearest sample
- Logistic regression does not guarantee the largest margin
- Drop the constraint of normalization for $\tilde{w}$ (i.e. $||w|| = 1$) and reformulate the signed distance using $\tilde{w}’ = \frac{\tilde{w}}{||w||}$

$$h = \tilde{w}' \cdot \tilde{x}=\frac{\tilde{w} \cdot \tilde{x}}{||w||}, \quad \forall \tilde{w} \in \mathbb{R}^{N+1}$$

## Maximum Margin Classifier
- Suppose a solution such that all the points are correctly classified : $\forall n \quad t_n(\tilde{\mathbf{w}} \cdot \tilde{\mathbf{x}_n}) \geq 0$
- **Unsigned** distance to the decision boundary is :

$$d_n = t_n\frac{(\tilde{\mathbf{w}} \cdot \tilde{\mathbf{x}}_n)}{||\mathbf{w}||}$$

- Maximizing the distance for the closest point to the boundary :

$$\tilde{\mathbf{w}}^* = \max_{\tilde{\mathbf{w}}} \min_n (\frac{t_n(\tilde{\mathbf{w}} \cdot \tilde{\mathbf{x}}_n)}{||\mathbf{w}||})$$

- Signed distance is invariant to a scaling $\lambda \not = 0$ of $\tilde{w}$ :

$$\tilde{w} \to \lambda \tilde{w} : d_n=t_n\frac{(\lambda \tilde{w} \cdot \tilde{x}_n)}{||\lambda w||}=t_n\frac{\tilde{w} \cdot \tilde{x}_n}{||w||}$$

- Value of $\lambda$ can be chosen such that for the closest point to the boundary $m$ :

$$\exist m\quad t_m (\tilde{w} \cdot x_m)=1 \\ \forall n \quad t_n(\tilde{w} \cdot x_n) \geq 1$$

- Minimization problem becomes :

$$\min_nd_n=\min_n \frac{t_n(\tilde{w} \cdot x_n)}{||w||}=\frac{1}{||w||} \implies \max \frac{1}{||w||} \quad \text{(maximizing the margin)}$$

- Max-margin classifier is found using 

$$w^*=\argmin_w \frac{1}{2}||w||^2 \quad \forall n, t_n(\tilde{w} \cdot x_n) \geq 1$$

- Quadratic problem with linear constraint → **convex** → global mininum

## Overlapping classes
- Constraint on distances such that $\forall n \quad t_n(\tilde{w} \cdot \tilde{x}_n) \geq 1$ is rarely achievable in practice (i.e. data is linearly separable)
- Relax the constraint → often $\forall n$ cannot be satisfied
- Allow some of the constraints to violated, but as few as possible

## Slack variables
- Slack variable $\xi_n$ for each sample and the constraint is rewrited as $t_n(\tilde{w} \cdot x_n) \geq 1- \xi_n$ where $\xi_n \geq 0$ weakens the original constraints
- If $0 < \xi_n \leq 1$ the sample $n$ lies inside the margin, but is still correctly classfied
- If $\xi_n \geq 1$ the sample $n$ is misclassified
- Add a **cost of error** $C$ to the objective function :

$$
w^*=\argmin_w \frac{1}{2} ||w||^2 + C\sum_{n=1}^N\xi_n \quad \forall n \quad t_n(\tilde{w} \cdot x_n) \geq 1 - \xi_n \quad \xi_n \geq 0 
$$

- Hyperparameter $C$ controls how costly constraint violations are
- Still convex
- Lower $C$ allows more misclassifications and hence lead to larger margins, while bigger $C$ reduces misclassfications and hence lead to smaller margins (trade-off)
- A larger margin is more robust and will better handle outliers
- A support vector is a data sample that is either on the margin or within the margin (or misclassified)

&nbsp;
&nbsp;
--- 
&nbsp;
&nbsp;

# Boosting : Adaboost (Ensemble method) (Non-Linear Supervised Classification + Regression)
- Combines weak learners into a strong learner that can make accurate predictions
- Logistic regression and other previous methods work well on linearly separable data with an hyperplane as decision boundary (linear classifier)
- Non linearly separable → combine multiple linear classifiers
- Approximates a non-linear decision boundary with a **strong** classifier that is a weighted sum of weak classifiers
- Adaboost is sensitive to noisy data and outliers, which can negatively impact its performance

$$y(\mathbf{x}) = \alpha_1 y_1(\mathbf{x}) + \alpha_2 y_2(\mathbf{x}) + \dots$$

## Weak classifier
- Classifier that only need to operate **slightly better than chance** (be right at least 50% of the time)
- Can be any linear classifiers we used before (Logistic Regression, SVM, ...) but can also be a non-linear classifier (e.g. boxes)

## Algorithm
- Algorithm assign weights to each sample
- At each iteration, it add a weak classifier and increase the weights of misclassified samples. Thus, the next iteration will focus on the points that are misclassified
- Initialize data weights $\forall n, w_n^1 = \frac{1}{N}$
- For $t \in [1, \dots, T]$ :
    - Find a weak classifier $y_t : \mathcal{X} \to \{-1, 1\}$ that minimizes weighted error $\sum_{t_n \not = y_t(\mathbf{x}_n)}w_n^t$
    - Evaluate :
        $$\epsilon_t = \frac{\sum_{t_n \not = y_t(\mathbf{x}_n)}w_n^t}{\sum_{n=1}^N w_n^t}$$
        $$\alpha_t = \log(\frac{1 - \epsilon_t}{\epsilon_t})$$ 
    - Update weights $w_n^{t+1} = w_n^t \exp(\alpha_t I(t_n \not = y_n(\mathbf{x}_n)))$


- The final **strong** classifier is : $y(\mathbf{x}) = sign(\sum^{T}_{t=1} \alpha_t y_t(\mathbf{x}))$
- Average weighted error $\epsilon_t$ of $y_t$ is $<0.5$ if $y_t$ operates at better than chance (and $\alpha_t > 0$)
- Number of iterations $T$ (i.e. number of weak classifiers) is a **hyperparameter**

## Training and Testing Errors

- Training error goes down exponentially fast if the weighted errors $\epsilon_t$ of the component classifiers is always $< 0.5$ (weak classifier consistenly do better than chance)

$$
\frac{1}{N}\sum_n[t_n \not=h(X_n)]<\prod_{t=1}^T\sqrt{\epsilon_t(1-\epsilon_t)}
$$

&nbsp;
&nbsp;
--- 
&nbsp;
&nbsp;

# Polynomial and RBF SVM (Non-Linear Supervised Classification)

- Data not linearly separable → use Adaboost to approximate the decision boundary, but in some cases it is not possible
- Another solution is to map data to a higher dimension

## Example : 2D Classification

- $\mathbf{x} = \begin{bmatrix}x_1 \\ x_2\end{bmatrix} \to \phi(\mathbf{x}) = \begin{bmatrix}x_1 \\ x_2 \\ x_1^2 + x_2^2\end{bmatrix}$ and use a linear classifier (hyperplane)

## Polynomial Approximation
- Given $(x_n ,t_n) \quad 1 \leq n \leq N$ we search for $f$ such that $t_n = f(x_n) + \epsilon$ (noise)
- Find $\mathbf{w}$ such that $\forall x, f(x) \approx \sum_{i=0}^M w_i x^i$
- The least squares solution is the most optimal
$$x \to \phi(x)=\begin{bmatrix}1 \\ x \\ x^2 \\ \vdots \\ x^M\end{bmatrix}$$
$$\mathbf{w}^* = \min_{\mathbf{w}} \sum_n (t_n - \sum_{i=0}^M w_i x_n^i) ^2 = \min_{\mathbf{w}} \sum_n (t_n - \mathbf{w}^T \phi(x_n))^2 = \min_\mathbf w ||\Phi\mathbf{w}-\mathbf{t}||^2$$

$$\Phi = 
\begin{bmatrix}
\phi(\mathbf{x}_1)^T \\
\phi(\mathbf{x}_2)^T \\
\vdots \\ 
\phi(\mathbf{x}_N)^T
\end{bmatrix} =
\begin{bmatrix}
1 & x_1 & x_1^2 & \dots & x_1^M \\
1 & x_2 & x_2^2 & \dots & x_2^M \\
\vdots \\ 
1 & x_N & x_N^2 & \dots & x_N^M
\end{bmatrix}
\quad \quad
\mathbf{w} = \begin{bmatrix}
w_0 \\
w_1 \\
\vdots \\ 
w_M
\end{bmatrix}
\quad \mathbf{t} = \begin{bmatrix}
t_0 \\
t_1 \\
\vdots \\ 
t_N
\end{bmatrix}$$

- Intutively it yields $\Phi \mathbf{w}^* \approx \mathbf{t}$, but more formally ($\nabla = 0$) :
$$(\Phi^T \Phi) \mathbf{w}^* = \Phi^T\mathbf{t}$$

## Regularization

$$\mathbf{w}^* = \argmin_\mathbf w ||\Phi\mathbf{w}-\mathbf{t}||^2 + \frac{\lambda}{2}||\mathbf{w}||^2 \\ \quad \\ (\Phi^T \Phi + \lambda \mathbf{I}) \mathbf{w}^* = \Phi^T\mathbf{t}$$

- Term with $\lambda$ is known as a **weight decay** → encourages the weights values to decay to zero unless supported by data (iterative algorithms)
- Discourages large weights and therefore quick variations that could lead to overfitting
- For linear and non-linear regression (e.g. polynomial approximation) → trade-off between simplicity and goodness of fit

## Classification in Feature Space
- Feature expansion function $\phi$ map data from $\mathbb{R}^d$ to $\mathbb{R}^D$.
- Output is computed similarly as in Logistic Regression but with the feature vector $\phi(\mathbf{x})$ 

$$y(\mathbf{x}) = \sigma(\mathbf{w}^T \phi(\mathbf{x}) + w_0)$$

## Polynomial Feature Expansion
- 1-Dimensional input expanded to degree $M$ yields a feature vector of dimension $M$
- $d$-Dimensional input, the dimension of $\phi(\mathbf{x})$ grows quickly with the degree $M$ of the polynomial

$$x \to \phi(x)=\begin{bmatrix}1 \\ x \\ x^2 \\ \vdots \\ x^M\end{bmatrix} \text{(1-D)}\quad \quad \quad \mathbf{x} \to \phi(\mathbf{x})=\begin{bmatrix}1 \\ x_1 \\ \vdots \\ x_d \\ x_1^2 \\ \vdots \\ x_d^2 \\ x_1^M \\ \vdots \\ x_d^M \\ x_1 x_2 \\ \vdots \end{bmatrix} \text{(d-D)}$$

- Feature vectors contain all possible monomials and polynomial up to degree $M$
- Can be used in any algorithm we seen (Perceptron, Logistic Regression, Linear SVM, Adaboost, ...)
- No theorical reason to choose polynomial expansion, but empirically it works well

## Polynomial SVM
- Most obvious way is to use the feature vector with linear SVM
- Polynomial SVM finds a linear SVM in higher dimension

$$
w^*=\argmin_{\mathbf{w}, \{\xi_n\}} \frac{1}{2} ||\mathbf{w}||^2 + C \sum_{n=1}^N \xi_n \\ \forall n \quad t_n(\tilde{\mathbf{w}} \cdot \phi(\mathbf{x}_n)) \geq 1 - \xi_n \quad \xi_n \geq 0 
$$

- Linear decision boundary in a high dimensional ($\mathbb{R}^D$) space becomes a curvy one in the original low dimensional space ($\mathbb{R}^d$)
- Higher-degree polynomial expansion yields a more flexible boundary and increase the chance that data become linearly separable
- A too flexible boundary can also results in overfitting and poor performance at test time
- It also increases the dimensionality of the problem → training SVMs is $O(D^3)$ → inherent limitation of polynomial SVMs

## Cover's Theorem
- Increasing the dimension of data also increase the probability that they will be linearly separable in the new space
- $\frac{C(p, N)}{2^p}$ is the percentage of separable partitions
- When $N$ is large, almost all partitions are separable if the number $p$ of samples is less than $2N$
- In practice, problems often contain billions of samples and therefore $N$ should be of that magnitude. Dealing with $N \times N$ matrices is impractical
- As the dimension increases, the boundaries become increasingly irregular and sensitive to noise
- Good news → world is structured and the points to classify are **not** randomly distributed (Cover's theorem makes the assumption that samples are randomly distributed in the whole space)
- Compute **feature vectors** that are *close* for objects that belong to the same class
- Non-linear SVM use the **Kernel Trick** method to increase the dimension while keeping the computational burden down

## Polynomial SVM without Slack Variables

$$\mathbf{w}^* = \min_{(\mathbf{w}, \{\xi_n\})} \frac{1}{2}||\mathbf{w}||^2 \quad \forall n, t_n \cdot (\tilde{\mathbf{w}} \cdot \phi(\mathbf{x}_n)) \geq 1$$

- Constrained optimization problem of the form : minimize $f(x, y)$ subject to $g(x, y) \leq c$
- At the constrained minimum the following equation holds :
$$\exists \lambda \in \mathbb{R} \quad \nabla f=\lambda \nabla g$$
- $\lambda$ is the **Lagrange multiplier** (one per constraint)
- Problem can be reformulated using the **Lagrangian** :
$$ L(\mathbf{w}, \Lambda)=\frac{1}{2}||\mathbf{w}||^2-\sum_{n=1}^N\lambda_n(t_n\tilde{\mathbf{w}}\cdot \phi(\mathbf{x}_n)-1) \\ \Lambda=[\lambda_1, \dots, \lambda_n] \quad \forall n, \lambda \geq 0$$
- Solution of the constrained minimization problem must be such that $L$ is minimized with respect to the components of $\mathbf{w}$ and maximized with respect to the Lagrange multipliers, which must remain $\geq 0$
- Settings the derivatives of $L(\mathbf{w}, \Lambda)$ to $0$ w.r.t. the elements of $\mathbf{w}$ and the sum yields :
$$\mathbf{w} = \sum_{n=1}^N \lambda_n t_n \phi(\mathbf{x}_n) \quad \quad 0 = \sum_{n=1}^N \lambda_n t_n$$
$$\min \tilde{L}(\Lambda)=L(\mathbf{w}, \Lambda)=\sum_{n=1}^N \lambda_n - \frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N \lambda_n \lambda_m t_n t_m k(\mathbf{x}_n, \mathbf{x}_m)$$
- Subject to $\sum_{n=1}^N \lambda_n t_n = 0 \quad \forall n, \lambda_n \geq 0$ and with $k(\mathbf{x}, \mathbf{x}') = \phi(\mathbf{x})^T\phi(\mathbf{x}')$.
- Quadratic programming problem with $N$ variables and a complexity in $O(N^3)$ instead of $O(D^3)$

## Support Vectors
$$\mathbf{w} = \sum_{n=1}^N \lambda_n t_n \phi(\mathbf{x}_n)$$
$$y(\mathbf{x})=\mathbf{w}^T\phi(\mathbf{x})+b = \sum_{n=1}^N \lambda_n t_n k(\mathbf{x}, \mathbf{x}_n) + b$$
- $\lambda_n$ are $\not = 0$ only for a subset of the samples → corresponding $\mathbf{x}_n$ are the **support vectors** and satisfy $t_ny(\mathbf{x}_n) = 1$
- Those samples are the only ones that need to be considered at test time and this fact is what makes SVMs practical
- There are as many *non-zero* $\xi_n$ as support vectors

## Test time
$$y(\mathbf{x})= \sum_{n \in \mathcal{S}}^N \lambda_n t_n k(\mathbf{x}, \mathbf{x}_n) + b$$
- $\mathcal S$ is the subset of samples where $\lambda_n \not = 0$
- Feature vector $\phi(\mathbf{x})$ does **not** appear explicitly anymore, instead it is replaced by the **kernel** function
- **Kernel function** can be understood as a similarity measure

## Kernel Trick
- Process of using $\phi$ implicitly (never computed in practice) by replacing it with the kernel $k$ (computed) is known as the *Kernel Trick* (used in many different algorithms besides SVMs)

$$
k(\mathbf{x}, \mathbf{x}') = ( \gamma \langle \mathbf{x}, \mathbf{x'} \rangle + r)^d  \quad \text{Polynomial}
$$

$$
k(\mathbf{x}, \mathbf{x}') = \exp(-\frac{||\mathbf{x} - \mathbf{x}'||^2}{\sigma^2}) \quad \text{RBF or Gaussian}
$$

- Gaussian kernel virtually makes the dimension as large as the number of samples
- Complexity now depends on $N$ and not the dimension $D$ anymore
- The smaller the $\sigma$ hyperparameter, the more complex the boundary with small "islands" (may lead to overfitting)
- Still not ideal for very large database where $N$ is high due to the $O(N^3)$
- Polynomial feature expansion explicitly computes the new features $\phi(\mathbf{x}_i)$ and performs linear SVM in the new feature space
-  Polynomial kernel function uses a function $k(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^\top\phi(\mathbf{x}_j)$ to compute a scalar product between two data points in the feature space, without having to explicitly send them there
- SVM trained with linear kernel on polynomially expanded data and SVM trained with polynomial kernel function on original data are similar. There might be small differences based on how the polynomial kernel function is written, or how the polynomial feature expansion is done, but the results should be equivalent

## Summary
- Data can be separable in a high-dimensional feature space without being separable in the input space
- Classifiers can be learned in the feature space without having to actually perform the mapping $\phi$
- $O(D^3)$ or $O(N^3)$ complexity at training time makes it hard to exploit large training sets → Deep Learning for large database

&nbsp;
&nbsp;
--- 
&nbsp;
&nbsp;

# Decision Trees and Forests

- Training set $\mathbf{v}$, to each sample $v$ is assigned a class $c$
- Each node has a weak learner $h(v, \theta) \in \{0, 1\}$ or (*false*, *true*)
- If the decision of the weak learner is *false*, the sample goes to the left child, else in the right one
- Decision forests make it comparatively easy to interpret what is happening
- Their behavior is easy to modify
- They can be trained using moderate amounts of data

## Training

- Proportion $p_l(c)$ of samples in each class that lands in leaf $l$ is computed

## Testing

- New sample $v \not \in \mathbf{v}$, we let $v$ flows through the tree and falls into leaf $l$
- Probability of belonging to class $c$ is $p(c|v) = p_l(c)$.

## Entropy and Gini Index
- $p^k$ is the proportion of data points in $\mathcal S$ that are assigned to class $k$
- The Gini Index $Q(\mathcal S) = \sum_{k=1}^Kp^k(1-p^k)$
- The entropy $Q(\mathcal S)=-\sum_{k=1}^Kp^k\ln p^k$
- Both vanish when $\exists k, p^k=1$ and they are maximized when all $p^k$ are equal
- Minimizing these measures favors leaves in which a large fraction of samples belong to the same class

## Maximizing information Gain

$$\max_{h(\mathbf{v}, \mathbf{\theta})} \quad Q(\mathcal S) - \sum_{\tau \in L, R} \frac{|\mathcal S^\tau|}{|\mathcal S|}Q(\mathcal S^\tau)$$

- For each node, pick the weak learner that delivers the highest information gain

## Forests
- Use multiple trees to increase robustness
$$
p(c|\textbf{v})=f(p_1(c|\textbf{v}), \dots, p_T(c|\textbf{v}))
$$

- Training sets of each tree $t$ are randomly sampled subsets $\mathcal S_0^t \sub \mathcal S_0$ of the full training set $\mathcal S_0$
- Subsets are typically chosen randomly with replacement : **bagging**
- Optimal numbers of trees is found using validation set
- Outputs can be combined using a Naive Bayesian approach : 
$$
p(c|\mathbf{v}) \propto \prod_t p_t(c|\mathbf{v}) \\ 

L(c, \mathbf{v}) = \frac{1}{T} \sum_t -log(p_t(c | \mathbf{v}))
$$
- $L$ is the negative likelihood (more numerically stable)
- Assumes that the ouput of each tree is independent from each other → valid assumption if the training subsets are disjoint
- Not the case with bagging (subsets with replacement), but it's ok if the training database is large enough

## Relationship to Boosting
- Boosting techniques produce very unbalanced tree  put appart some samples at each level and continues with the other)
- *Boosted cascades* are good for unbalanced binary problems, such as sliding window object detection (e.g. face detection) (far more windows with nothing that with faces)
- Randomized forests are less deep and more balanced
- Ensemble of trees gives robustness and are good for multi-class problems
