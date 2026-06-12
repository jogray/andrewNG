#!/usr/bin/env python3
"""Translate English markdown and comments in Jupyter notebooks to Chinese."""

import json
import re

# Translation mapping for markdown cells (English -> Chinese)
# We'll use line-by-line translation for code comments
# and block-level translation for markdown cells

def translate_markdown_line(line):
    """Translate a single markdown line from English to Chinese."""
    # Preserve HTML tags, LaTeX, links etc.
    translations = [
        # Title and outline
        ("# Logistic Regression", "# 逻辑回归"),
        ("# Outline", "# 大纲"),
        
        # Section 1 - Packages
        ("## 1 - Packages", "## 1 - 软件包"),
        ("First, let's run the cell below to import all the packages that you will need during this assignment.",
         "首先，让我们运行下面的单元格来导入本次作业所需的所有软件包。"),
        ("is the fundamental package for scientific computing with Python.",
         "是Python科学计算的基础软件包。"),
        ("is a famous library to plot graphs in Python.",
         "是Python中著名的绘图库。"),
        ("contains helper functions for this assignment. You do not need to modify code in this file.",
         "包含本次作业的辅助函数。您不需要修改此文件中的代码。"),
        
        # Section 2 - Logistic Regression
        ("## 2 - Logistic Regression", "## 2 - 逻辑回归"),
        ("In this part of the exercise, you will build a logistic regression model to predict whether a student gets admitted into a university.",
         "在本部分练习中，您将构建一个逻辑回归模型来预测学生是否被大学录取。"),
        
        # 2.1 Problem Statement
        ("### 2.1 Problem Statement", "### 2.1 问题描述"),
        ("Suppose that you are the administrator of a university department and you want to determine each applicant's chance of admission based on their results on two exams.",
         "假设您是一个大学部门的管理员，您想根据申请人在两次考试中的成绩来确定每个申请人的录取机会。"),
        ("You have historical data from previous applicants that you can use as a training set for logistic regression.",
         "您有以往申请人的历史数据，可以作为逻辑回归的训练集。"),
        ("For each training example, you have the applicant's scores on two exams and the admissions decision.",
         "对于每个训练样本，您有申请人在两次考试中的分数和录取决定。"),
        ("Your task is to build a classification model that estimates an applicant's probability of admission based on the scores from those two exams.",
         "您的任务是构建一个分类模型，根据这两次考试的分数来估计申请人的录取概率。"),
        
        # 2.2 Loading and visualizing the data
        ("### 2.2 Loading and visualizing the data", "### 2.2 加载和可视化数据"),
        ("You will start by loading the dataset for this task.",
         "您将首先加载此任务的数据集。"),
        ("The `load_dataset()` function shown below loads the data into variables `X_train` and `y_train`",
         "下面显示的 `load_dataset()` 函数将数据加载到变量 `X_train` 和 `y_train` 中"),
        ("`X_train` contains exam scores on two exams for a student",
         "`X_train` 包含学生两次考试的成绩"),
        ("`y_train` is the admission decision",
         "`y_train` 是录取决定"),
        ("if the student was admitted", "如果学生被录取"),
        ("if the student was not admitted", "如果学生未被录取"),
        ("Both `X_train` and `y_train` are numpy arrays.", "`X_train` 和 `y_train` 都是numpy数组。"),
        
        # View variables
        ("#### View the variables", "#### 查看变量"),
        ("Let's get more familiar with your dataset.", "让我们更熟悉您的数据集。"),
        ("A good place to start is to just print out each variable and see what it contains.",
         "一个好的起点是打印出每个变量，看看它包含什么。"),
        ("The code below prints the first five values of `X_train` and the type of the variable.",
         "下面的代码打印 `X_train` 的前五个值和变量的类型。"),
        ("Now print the first five values of `y_train`", "现在打印 `y_train` 的前五个值"),
        
        # Check dimensions
        ("#### Check the dimensions of your variables", "#### 检查变量的维度"),
        ("Another useful way to get familiar with your data is to view its dimensions. Let's print the shape of `X_train` and `y_train` and see how many training examples we have in our dataset.",
         "另一种了解数据的有用方法是查看其维度。让我们打印 `X_train` 和 `y_train` 的形状，看看数据集中有多少训练样本。"),
        
        # Visualize data
        ("#### Visualize your data", "#### 可视化数据"),
        ("Before starting to implement any learning algorithm, it is always good to visualize the data if possible.",
         "在开始实现任何学习算法之前，如果可能的话，最好先可视化数据。"),
        ("The code below displays the data on a 2D plot (as shown below), where the axes are the two exam scores, and the positive and negative examples are shown with different markers.",
         "下面的代码将数据显示在二维图上（如下所示），其中坐标轴是两次考试的分数，正例和负例用不同的标记显示。"),
        ("We use a helper function in the", "我们使用中的辅助函数"),
        ("file to generate this plot.", "文件来生成此图。"),
        
        # Goal
        ("Your goal is to build a logistic regression model to fit this data.",
         "您的目标是构建一个逻辑回归模型来拟合这些数据。"),
        ("With this model, you can then predict if a new student will be admitted based on their scores on the two exams.",
         "使用此模型，您可以根据新学生在两次考试中的分数来预测其是否会被录取。"),
        
        # 2.3 Sigmoid function
        ("### 2.3  Sigmoid function", "### 2.3 Sigmoid函数"),
        ("Recall that for logistic regression, the model is represented as",
         "回顾一下，对于逻辑回归，模型表示为"),
        ("where function $g$ is the sigmoid function. The sigmoid function is defined as:",
         "其中函数 $g$ 是sigmoid函数。Sigmoid函数定义为："),
        ("Let's implement the sigmoid function first, so it can be used by the rest of this assignment.",
         "让我们先实现sigmoid函数，以便在本作业的其余部分中使用。"),
        
        # Exercise 1
        ("### Exercise 1", "### 练习1"),
        ("Please complete  the `sigmoid` function to calculate",
         "请完成 `sigmoid` 函数来计算"),
        ("Note that", "注意"),
        ("`z` is not always a single number, but can also be an array of numbers.",
         "`z` 不总是一个数字，也可以是一个数字数组。"),
        ("If the input is an array of numbers, we'd like to apply the sigmoid function to each value in the input array.",
         "如果输入是一个数字数组，我们希望对输入数组中的每个值应用sigmoid函数。"),
        ("If you get stuck, you can check out the hints presented after the cell below to help you with the implementation.",
         "如果您遇到困难，可以查看下面单元格后提供的提示来帮助您实现。"),
        
        # Hints
        ("Click for hints", "点击查看提示"),
        ("Click for more hints", "点击查看更多提示"),
        ("Hint to calculate g", "计算g的提示"),
        ("You can translate", "您可以将"),
        ("into code as", "转换为代码"),
        
        # Testing sigmoid
        ("When you are finished, try testing a few values by calling `sigmoid(x)` in the cell below.",
         "完成后，尝试在下面的单元格中调用 `sigmoid(x)` 来测试几个值。"),
        ("For large positive values of x, the sigmoid should be close to 1, while for large negative values, the sigmoid should be close to 0.",
         "对于较大的正值x，sigmoid应接近1，而对于较大的负值，sigmoid应接近0。"),
        ("Evaluating `sigmoid(0)` should give you exactly 0.5.",
         "计算 `sigmoid(0)` 应该恰好得到0.5。"),
        
        # Expected output
        ("**Expected Output**:", "**预期输出**："),
        ("As mentioned before, your code should also work with vectors and matrices. For a matrix, your function should perform the sigmoid function on every element.",
         "如前所述，您的代码也应该适用于向量和矩阵。对于矩阵，您的函数应该对每个元素执行sigmoid函数。"),
        
        # 2.4 Cost function
        ("### 2.4 Cost function for logistic regression", "### 2.4 逻辑回归的代价函数"),
        ("In this section, you will implement the cost function for logistic regression.",
         "在本节中，您将实现逻辑回归的代价函数。"),
        ("### Exercise 2", "### 练习2"),
        ("Please complete the `compute_cost` function using the equations below.",
         "请使用下面的公式完成 `compute_cost` 函数。"),
        ("Recall that for logistic regression, the cost function is of the form",
         "回顾一下，对于逻辑回归，代价函数的形式为"),
        ("where", "其中"),
        ("m is the number of training examples in the dataset", "m是数据集中训练样本的数量"),
        ("is the cost for a single data point, which is -", "是单个数据点的代价，即-"),
        ("is the model's prediction, while $y^{(i)}$, which is the actual label",
         "是模型的预测值，而 $y^{(i)}$ 是实际标签"),
        ("where function $g$ is the sigmoid function.", "其中函数 $g$ 是sigmoid函数。"),
        ("It might be helpful to first calculate an intermediate variable",
         "首先计算一个中间变量可能会有所帮助"),
        ("where $n$ is the number of features, before calculating",
         "其中 $n$ 是特征数量，然后再计算"),
        ("As you are doing this, remember that the variables `X_train` and `y_train` are not scalar values but matrices of shape",
         "在执行此操作时，请记住变量 `X_train` 和 `y_train` 不是标量值，而是形状为"),
        ("where  $𝑛$ is the number of features and $𝑚$ is the number of training examples.",
         "的矩阵，其中 $𝑛$ 是特征数量，$𝑚$ 是训练样本数量。"),
        ("You can use the sigmoid function that you implemented above for this part.",
         "您可以使用上面实现的sigmoid函数来完成此部分。"),
        
        # Hints for cost function
        ("Here's how you can structure the overall implementation for this function",
         "以下是您如何构建此函数的整体实现"),
        ("You can represent a summation operator", "您可以将求和运算符"),
        ("in code as follows:", "用代码表示如下："),
        ("In this case, you can iterate over all the examples in `X` using a for loop and add the `loss` from each iteration to a variable",
         "在这种情况下，您可以使用for循环遍历 `X` 中的所有样本，并将每次迭代的 `loss` 添加到一个变量中"),
        ("initialized outside the loop.", "在循环外部初始化。"),
        ("Then, you can return the `total_cost` as `loss_sum` divided by `m`.",
         "然后，您可以将 `total_cost` 返回为 `loss_sum` 除以 `m`。"),
        ("Hint to calculate z_wb_ij", "计算z_wb_ij的提示"),
        ("Hint to calculate f_wb", "计算f_wb的提示"),
        ("is the sigmoid function. You can simply call the `sigmoid` function implemented above.",
         "是sigmoid函数。您可以简单地调用上面实现的 `sigmoid` 函数。"),
        ("More hints to calculate f", "计算f的更多提示"),
        ("You can compute f_wb as", "您可以将f_wb计算为"),
        ("Hint to calculate loss", "计算loss的提示"),
        ("You can use the", "您可以使用"),
        ("function to calculate the log", "函数来计算对数"),
        ("More hints to calculate loss", "计算loss的更多提示"),
        ("You can compute loss as", "您可以将loss计算为"),
        
        # Run cost check
        ("Run the cells below to check your implementation of the `compute_cost` function with two different initializations of the parameters $w$",
         "运行下面的单元格，使用参数 $w$ 的两种不同初始化来检查您实现的 `compute_cost` 函数"),
        
        # 2.5 Gradient
        ("### 2.5 Gradient for logistic regression", "### 2.5 逻辑回归的梯度"),
        ("In this section, you will implement the gradient for logistic regression.",
         "在本节中，您将实现逻辑回归的梯度。"),
        ("Recall that the gradient descent algorithm is:", "回顾一下，梯度下降算法是："),
        ("where, parameters $b$, $w_j$ are all updated simultaniously",
         "其中，参数 $b$、$w_j$ 都是同时更新的"),
        
        # Exercise 3
        ("### Exercise 3", "### 练习3"),
        ("Please complete the `compute_gradient` function to compute",
         "请完成 `compute_gradient` 函数来计算"),
        ("from equations (2) and (3) below.", "根据下面的公式(2)和(3)。"),
        ("m is the number of training examples in the dataset", "m是数据集中训练样本的数量"),
        ("is the model's prediction, while $y^{(i)}$ is the actual label",
         "是模型的预测值，而 $y^{(i)}$ 是实际标签"),
        ("**Note**: While this gradient looks identical to the linear regression gradient, the formula is actually different because linear and logistic regression have different definitions of",
         "**注意**：虽然这个梯度看起来与线性回归梯度相同，但公式实际上是不同的，因为线性回归和逻辑回归对"),
        ("has different definitions of", "有不同的定义"),
        ("As before, you can use the sigmoid function that you implemented above and if you get stuck, you can check out the hints presented after the cell below to help you with the implementation.",
         "和之前一样，您可以使用上面实现的sigmoid函数，如果您遇到困难，可以查看下面单元格后提供的提示来帮助您实现。"),
        
        # Hints for gradient
        ("Run the cells below to check your implementation of the `compute_gradient` function with two different initializations of the parameters $w$",
         "运行下面的单元格，使用参数 $w$ 的两种不同初始化来检查您实现的 `compute_gradient` 函数"),
        ("Hint to calculate dj_db_i", "计算dj_db_i的提示"),
        ("Hint to calculate dj_dw_ij", "计算dj_dw_ij的提示"),
        
        # 2.6 Gradient descent
        ("### 2.6 Learning parameters using gradient descent", "### 2.6 使用梯度下降学习参数"),
        ("Similar to the previous assignment, you will now find the optimal parameters of a logistic regression model by using gradient descent.",
         "与之前的作业类似，您现在将使用梯度下降来找到逻辑回归模型的最优参数。"),
        ("You don't need to implement anything for this part. Simply run the cells below.",
         "此部分不需要您实现任何内容。只需运行下面的单元格。"),
        ("A good way to verify that gradient descent is working correctly is to look",
         "验证梯度下降是否正确工作的一个好方法是查看"),
        ("at the value of $J(\\mathbf{w},b)$ and check that it is decreasing with each step.",
         "$J(\\mathbf{w},b)$ 的值，并检查它是否在每一步都在减小。"),
        ("Assuming you have implemented the gradient and computed the cost correctly, your value of $J(\\mathbf{w},b)$ should never increase, and should converge to a steady value by the end of the algorithm.",
         "假设您已正确实现梯度并计算了代价，您的 $J(\\mathbf{w},b)$ 值应该永远不会增加，并且应该在算法结束时收敛到一个稳定值。"),
        
        ("Now let's run the gradient descent algorithm above to learn the parameters for our dataset.",
         "现在让我们运行上面的梯度下降算法来学习我们数据集的参数。"),
        ("**Note**", "**注意**"),
        ("The code block below takes a couple of minutes to run, especially with a non-vectorized version. You can reduce the `iterations` to test your implementation and iterate faster. If you have time, try running 100,000 iterations for better results.",
         "下面的代码块需要几分钟来运行，特别是在非向量化版本中。您可以减少 `iterations` 来测试您的实现并更快地迭代。如果有时间，请尝试运行100,000次迭代以获得更好的结果。"),
        
        # Expected output for gradient descent
        ("Expected Output: Cost     0.30, (Click to see details):", "预期输出：代价 0.30，（点击查看详情）："),
        
        # 2.7 Decision boundary
        ("### 2.7 Plotting the decision boundary", "### 2.7 绘制决策边界"),
        ("We will now use the final parameters from gradient descent to plot the linear fit. If you implemented the previous parts correctly, you should see the following plot:",
         "我们现在将使用梯度下降的最终参数来绘制线性拟合。如果您正确实现了前面的部分，您应该看到以下图表："),
        ("We will use a helper function in the `utils.py` file to create this plot.",
         "我们将使用 `utils.py` 文件中的辅助函数来创建此图表。"),
        
        # 2.8 Evaluating
        ("### 2.8 Evaluating logistic regression", "### 2.8 评估逻辑回归"),
        ("We can evaluate the quality of the parameters we have found by seeing how well the learned model predicts on our training set.",
         "我们可以通过查看学习到的模型在训练集上的预测效果来评估我们找到的参数的质量。"),
        ("You will implement the `predict` function below to do this.", "您将在下面实现 `predict` 函数来完成此操作。"),
        
        # Exercise 4
        ("### Exercise 4", "### 练习4"),
        ("Please complete the `predict` function to produce `1` or `0` predictions given a dataset and a learned parameter vector $w$ and $b$.",
         "请完成 `predict` 函数，给定数据集和学习到的参数向量 $w$ 和 $b$，产生 `1` 或 `0` 的预测。"),
        ("First you need to compute the prediction from the model", "首先您需要从模型计算预测"),
        ("for every example", "对于每个样本"),
        ("You've implemented this before in the parts above", "您之前在上面的部分中已经实现了这个"),
        ("We interpret the output of the model", "我们将模型的输出解释为"),
        ("as the probability that $y^{(i)}=1$ given $x^{(i)}$ and parameterized by $w$.",
         "在给定 $x^{(i)}$ 和参数 $w$ 的条件下 $y^{(i)}=1$ 的概率。"),
        ("Therefore, to get a final prediction", "因此，要获得最终预测"),
        ("from the logistic regression model, you can use the following heuristic -",
         "从逻辑回归模型中，您可以使用以下启发式方法 -"),
        ("predict", "预测"),
        
        # Hints for predict
        ("Once you have completed the function `predict`, let's run the code below to report the training accuracy of your classifier by computing the percentage of examples it got correct.",
         "完成 `predict` 函数后，让我们运行下面的代码，通过计算分类器正确预测的样本百分比来报告其训练准确率。"),
        ("**Expected output**", "**预期输出**"),
        ("Now let's use this to compute the accuracy on the training set",
         "现在让我们用它来计算训练集上的准确率"),
        
        # Section 3 - Regularized Logistic Regression
        ("## 3 - Regularized Logistic Regression", "## 3 - 正则化逻辑回归"),
        ("In this part of the exercise, you will implement regularized logistic regression to predict whether microchips from a fabrication plant passes quality assurance (QA). During QA, each microchip goes through various tests to ensure it is functioning correctly.",
         "在本部分练习中，您将实现正则化逻辑回归来预测制造工厂的微芯片是否通过质量保证(QA)。在QA过程中，每个微芯片都会经过各种测试以确保其正常运行。"),
        
        # 3.1 Problem Statement
        ("### 3.1 Problem Statement", "### 3.1 问题描述"),
        ("Suppose you are the product manager of the factory and you have the test results for some microchips on two different tests.",
         "假设您是工厂的产品经理，您有一些微芯片在两项不同测试中的测试结果。"),
        ("From these two tests, you would like to determine whether the microchips should be accepted or rejected.",
         "根据这两项测试，您想确定微芯片应该被接受还是拒绝。"),
        ("To help you make the decision, you have a dataset of test results on past microchips, from which you can build a logistic regression model.",
         "为了帮助您做出决定，您有一个过去微芯片测试结果的数据集，可以从中构建逻辑回归模型。"),
        
        # 3.2 Loading and visualizing
        ("### 3.2 Loading and visualizing the data", "### 3.2 加载和可视化数据"),
        ("Similar to previous parts of this exercise, let's start by loading the dataset for this task and visualizing it.",
         "与本练习的前几部分类似，让我们首先加载此任务的数据集并对其进行可视化。"),
        ("`X_train` contains the test results for the microchips from two tests",
         "`X_train` 包含微芯片两项测试的结果"),
        ("`y_train` contains the results of the QA",
         "`y_train` 包含QA的结果"),
        ("if the microchip was accepted", "如果微芯片被接受"),
        ("if the microchip was rejected", "如果微芯片被拒绝"),
        
        # View variables for section 3
        ("The code below prints the first five values of `X_train` and `y_train` and the type of the variables.",
         "下面的代码打印 `X_train` 和 `y_train` 的前五个值以及变量的类型。"),
        
        # Visualize for section 3
        ("The helper function `plot_data` (from `utils.py`) is used to generate a figure like Figure 3, where the axes are the two test scores, and the positive (y = 1, accepted) and negative (y = 0, rejected) examples are shown with different markers.",
         "辅助函数 `plot_data`（来自 `utils.py`）用于生成类似图3的图表，其中坐标轴是两项测试的分数，正例（y = 1，接受）和负例（y = 0，拒绝）用不同的标记显示。"),
        ("Figure 3 shows that our dataset cannot be separated into positive and negative examples by a straight-line through the plot. Therefore, a straight forward application of logistic regression will not perform well on this dataset since logistic regression will only be able to find a linear decision boundary.",
         "图3显示我们的数据集无法通过图中的直线将正例和负例分开。因此，直接应用逻辑回归在此数据集上表现不佳，因为逻辑回归只能找到线性决策边界。"),
        
        # 3.3 Feature mapping
        ("### 3.3 Feature mapping", "### 3.3 特征映射"),
        ("One way to fit the data better is to create more features from each data point. In the provided function `map_feature`, we will map the features into all polynomial terms of $x_1$ and $x_2$ up to the sixth power.",
         "更好地拟合数据的一种方法是从每个数据点创建更多特征。在提供的函数 `map_feature` 中，我们将特征映射为 $x_1$ 和 $x_2$ 的所有多项式项，最高到六次幂。"),
        ("As a result of this mapping, our vector of two features (the scores on two QA tests) has been transformed into a 27-dimensional vector.",
         "通过此映射，我们的两个特征向量（两次QA测试的分数）已被转换为27维向量。"),
        ("A logistic regression classifier trained on this higher-dimension feature vector will have a more complex decision boundary and will be nonlinear when drawn in our 2-dimensional plot.",
         "在此高维特征向量上训练的逻辑回归分类器将具有更复杂的决策边界，在我们的二维图中绘制时将是非线性的。"),
        ("We have provided the `map_feature` function for you in utils.py.",
         "我们已在utils.py中为您提供了 `map_feature` 函数。"),
        ("Let's also print the first elements of `X_train` and `mapped_X` to see the tranformation.",
         "让我们也打印 `X_train` 和 `mapped_X` 的第一个元素来查看转换。"),
        
        # 3.4 Cost function for regularized
        ("While the feature mapping allows us to build a more expressive classifier, it is also more susceptible to overfitting. In the next parts of the exercise, you will implement regularized logistic regression to fit the data and also see for yourself how regularization can help combat the overfitting problem.",
         "虽然特征映射允许我们构建更具表达力的分类器，但它也更容易过拟合。在练习的接下来的部分中，您将实现正则化逻辑回归来拟合数据，并亲自了解正则化如何帮助解决过拟合问题。"),
        ("### 3.4 Cost function for regularized logistic regression", "### 3.4 正则化逻辑回归的代价函数"),
        ("In this part, you will implement the cost function for regularized logistic regression.",
         "在本部分中，您将实现正则化逻辑回归的代价函数。"),
        ("Recall that for regularized logistic regression, the cost function is of the form",
         "回顾一下，对于正则化逻辑回归，代价函数的形式为"),
        ("Compare this to the cost function without regularization (which you implemented above), which is of the form",
         "将其与没有正则化的代价函数（您在上面实现的）进行比较，其形式为"),
        ("The difference is the regularization term, which is", "区别在于正则化项，即"),
        ("Note that the $b$ parameter is not regularized.", "注意 $b$ 参数没有被正则化。"),
        
        # Exercise 5
        ("### Exercise 5", "### 练习5"),
        ("Please complete the `compute_cost_reg` function below to calculate the following term for each element in $w$",
         "请完成下面的 `compute_cost_reg` 函数来计算 $w$ 中每个元素的以下项"),
        ("The starter code then adds this to the cost without regularization (which you computed above in `compute_cost`) to calculate the cost with regulatization.",
         "初始代码将其添加到没有正则化的代价（您在上面的 `compute_cost` 中计算的）中，以计算带正则化的代价。"),
        
        # Hints for regularized cost
        ("Run the cell below to check your implementation of the `compute_cost_reg` function.",
         "运行下面的单元格来检查您实现的 `compute_cost_reg` 函数。"),
        ("Hint to calculate reg_cost_j", "计算reg_cost_j的提示"),
        
        # 3.5 Gradient for regularized
        ("### 3.5 Gradient for regularized logistic regression", "### 3.5 正则化逻辑回归的梯度"),
        ("In this section, you will implement the gradient for regularized logistic regression.",
         "在本节中，您将实现正则化逻辑回归的梯度。"),
        ("The gradient of the regularized cost function has two components. The first,",
         "正则化代价函数的梯度有两个分量。第一个，"),
        ("is a scalar, the other is a vector with the same shape as the parameters $\\mathbf{w}$, where the $j^\\mathrm{th}$ element is defined as follows:",
         "是一个标量，另一个是与参数 $\\mathbf{w}$ 形状相同的向量，其中第 $j$ 个元素定义如下："),
        ("Compare this to the gradient of the cost function without regularization (which you implemented above), which is of the form",
         "将其与没有正则化的代价函数的梯度（您在上面实现的）进行比较，其形式为"),
        ("As you can see,", "如您所见，"),
        ("is the same, the difference is the following term in", "是相同的，区别在于以下项在"),
        ("which is", "即"),
        
        # Exercise 6
        ("### Exercise 6", "### 练习6"),
        ("Please complete the `compute_gradient_reg` function below to modify the code below to calculate the following term",
         "请完成下面的 `compute_gradient_reg` 函数来修改下面的代码以计算以下项"),
        ("The starter code will add this term to the",
         "初始代码将把这个项添加到"),
        ("returned from `compute_gradient` above to get the gradient for the regularized cost function.",
         "从上面的 `compute_gradient` 返回的值中，以获得正则化代价函数的梯度。"),
        
        # Hints for regularized gradient
        ("Run the cell below to check your implementation of the `compute_gradient_reg` function.",
         "运行下面的单元格来检查您实现的 `compute_gradient_reg` 函数。"),
        ("Hint to calculate dj_dw_j_reg", "计算dj_dw_j_reg的提示"),
        
        # 3.6 Learning parameters
        ("### 3.6 Learning parameters using gradient descent", "### 3.6 使用梯度下降学习参数"),
        ("Similar to the previous parts, you will use your gradient descent function implemented above to learn the optimal parameters $w$,$b$.",
         "与前面部分类似，您将使用上面实现的梯度下降函数来学习最优参数 $w$,$b$。"),
        ("If you have completed the cost and gradient for regularized logistic regression correctly, you should be able to step through the next cell to learn the parameters $w$.",
         "如果您正确完成了正则化逻辑回归的代价和梯度，您应该能够逐步执行下一个单元格来学习参数 $w$。"),
        ("After training our parameters, we will use it to plot the decision boundary.",
         "训练参数后，我们将使用它来绘制决策边界。"),
        ("The code block below takes quite a while to run, especially with a non-vectorized version. You can reduce the `iterations` to test your implementation and iterate faster. If you hae time, run for 100,000 iterations to see better results.",
         "下面的代码块需要相当长的时间来运行，特别是在非向量化版本中。您可以减少 `iterations` 来测试您的实现并更快地迭代。如果有时间，请运行100,000次迭代以获得更好的结果。"),
        
        # Expected output for regularized gradient descent
        ("Expected Output: Cost < 0.5  (Click for details)", "预期输出：代价 < 0.5（点击查看详情）"),
        
        # 3.7 Decision boundary for regularized
        ("### 3.7 Plotting the decision boundary", "### 3.7 绘制决策边界"),
        ("To help you visualize the model learned by this classifier, we will use our `plot_decision_boundary` function which plots the (non-linear) decision boundary that separates the positive and negative examples.",
         "为了帮助您可视化此分类器学习的模型，我们将使用 `plot_decision_boundary` 函数来绘制分隔正例和负例的（非线性）决策边界。"),
        ("In the function, we plotted the non-linear decision boundary by computing the classifier's predictions on an evenly spaced grid and then drew a contour plot of where the predictions change from y = 0 to y = 1.",
         "在函数中，我们通过计算分类器在均匀间隔网格上的预测来绘制非线性决策边界，然后绘制预测从y = 0变为y = 1的等高线图。"),
        ("After learning the parameters $w$,$b$, the next step is to plot a decision boundary similar to Figure 4.",
         "学习参数 $w$,$b$ 后，下一步是绘制类似图4的决策边界。"),
        
        # 3.8 Evaluating regularized
        ("### 3.8 Evaluating regularized logistic regression model", "### 3.8 评估正则化逻辑回归模型"),
        ("You will use the `predict` function that you implemented above to calculate the accuracy of the regulaized logistic regression model on the training set",
         "您将使用上面实现的 `predict` 函数来计算正则化逻辑回归模型在训练集上的准确率"),
        
        # NOTE block (for 20230213 file)
        ("_**NOTE:** To prevent errors from the autograder, you are not allowed to edit or delete non-graded cells in this lab. Please also refrain from adding any new cells.",
         "_**注意：** 为防止自动评分器出错，您不允许编辑或删除本实验中的非评分单元格。请也不要添加任何新单元格。"),
        ("**Once you have passed this assignment** and want to experiment with any of the non-graded code, you may follow the instructions at the bottom of this notebook._",
         "**通过此作业后**，如果您想试验任何非评分代码，可以按照本笔记本底部的说明进行操作。_"),
    ]
    
    result = line
    for eng, chn in translations:
        result = result.replace(eng, chn)
    return result


def translate_code_comment(line):
    """Translate English comments in Python code to Chinese."""
    # Only translate comments (lines starting with # or inline comments)
    # Do NOT translate code, strings, or print statements
    
    comment_translations = {
        "# load dataset": "# 加载数据集",
        "# Plot examples": "# 绘制样本",
        "# Set the y-axis label": "# 设置y轴标签",
        "# Set the x-axis label": "# 设置x轴标签",
        "# UNIT TESTS": "# 单元测试",
        "# Compute and display cost with w initialized to zeroes": "# 计算并显示w初始化为零时代的价",
        "# Compute and display cost with non-zero w": "# 计算并显示非零w时的代价",
        "# Compute and display gradient with w initialized to zeroes": "# 计算并显示w初始化为零时的梯度",
        "# Compute and display cost and gradient with non-zero w": "# 计算并显示非零w时的代价和梯度",
        "# Some gradient descent settings": "# 一些梯度下降设置",
        "# Initialize fitting parameters": "# 初始化拟合参数",
        "# Set regularization parameter lambda_ to 1 (you can try varying this)": "# 设置正则化参数lambda_为1（您可以尝试更改）",
        "# print X_train": "# 打印 X_train",
        "# print y_train": "# 打印 y_train",
        "#Compute accuracy on our training set": "# 计算训练集上的准确率",
        "#Compute accuracy on the training set": "# 计算训练集上的准确率",
        "# Test your predict code": "# 测试您的预测代码",
        "# Calls the compute_cost function that you implemented above": "# 调用您上面实现的compute_cost函数",
        "# You need to calculate this value": "# 您需要计算这个值",
        "# Add the regularization cost to get the total cost": "# 添加正则化代价以获得总代价",
        "# number of training examples": "# 训练样本数量",
        "# An array to store cost J and w's at each iteration primarily for graphing later": "# 一个数组，用于存储每次迭代的代价J和w，主要用于后续绘图",
        "# Calculate the gradient and update the parameters": "# 计算梯度并更新参数",
        "# Update Parameters using w, b, alpha and gradient": "# 使用w、b、alpha和梯度更新参数",
        "# Save cost J at each iteration": "# 保存每次迭代的代价J",
        "# prevent resource exhaustion": "# 防止资源耗尽",
        "# Print cost every at intervals 10 times or as many iterations if < 10": "# 每隔10次或少于10次时打印代价",
        "#return w and J,w history for graphing": "# 返回w和J,w历史用于绘图",
    }
    
    stripped = line.strip()
    if stripped in comment_translations:
        return line.replace(stripped, comment_translations[stripped])
    
    return line


def translate_docstring(content):
    """Translate docstring content from English to Chinese."""
    docstring_translations = {
        "Compute the sigmoid of z": "计算z的sigmoid值",
        "A scalar, numpy array of any size.": "任意大小的标量或numpy数组。",
        "sigmoid(z), with the same shape as z": "sigmoid(z)，与z形状相同",
        "Computes the cost over all examples": "计算所有样本的代价",
        "data, m examples by n features": "数据，m个样本，n个特征",
        "target value": "目标值",
        "Values of parameters of the model": "模型参数的值",
        "Values of bias parameter of the model": "模型偏置参数的值",
        "unused placeholder": "未使用的占位符",
        "Controls amount of regularization": "控制正则化的程度",
        "cost": "代价",
        "Computes the gradient for logistic regression": "计算逻辑回归的梯度",
        "variable such as house size": "变量，如房屋大小",
        "actual value": "实际值",
        "values of parameters of the model": "模型参数的值",
        "value of parameter of the model": "模型参数的值",
        "value of parameter of the model": "模型参数的值",
        "The gradient of the cost w.r.t. the parameters w.": "代价相对于参数w的梯度。",
        "The gradient of the cost w.r.t. the parameter b.": "代价相对于参数b的梯度。",
        "Performs batch gradient descent to learn theta. Updates theta by taking": "执行批量梯度下降来学习theta。通过执行",
        "num_iters gradient steps with learning rate alpha": "num_iters次梯度下降步骤，学习率为alpha",
        "Initial values of parameters of the model": "模型参数的初始值",
        "Initial value of parameter of the model": "模型参数的初始值",
        "function to compute cost": "计算代价的函数",
        "Learning rate": "学习率",
        "number of iterations to run gradient descent": "运行梯度下降的迭代次数",
        "regularization constant": "正则化常数",
        "Updated values of parameters of the model after": "模型参数的更新值，在",
        "running gradient descent": "运行梯度下降之后",
        "Updated value of parameter of the model after": "模型参数的更新值，在",
        "Predict whether the label is 0 or 1 using learned logistic": "使用学习到的逻辑回归参数w",
        "regression parameters w": "预测标签是0还是1",
        "The predictions for X using a threshold at 0.5": "使用0.5阈值对X的预测",
        "Computes the gradient for linear regression": "计算线性回归的梯度",
        "The gradient of the cost w.r.t. the parameters w.": "代价相对于参数w的梯度。",
    }
    
    result = content
    for eng, chn in docstring_translations.items():
        result = result.replace(eng, chn)
    return result


def translate_notebook(filepath):
    """Translate a single notebook file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            new_source = []
            for line in cell['source']:
                new_source.append(translate_markdown_line(line))
            cell['source'] = new_source
        
        elif cell['cell_type'] == 'code':
            new_source = []
            in_docstring = False
            for line in cell['source']:
                stripped = line.strip()
                
                # Translate docstrings
                if '"""' in stripped:
                    if stripped.count('"""') == 1:
                        in_docstring = not in_docstring
                        new_source.append(line)
                        continue
                    elif stripped.count('"""') == 2:
                        # Single line docstring
                        new_source.append(translate_docstring(line))
                        continue
                
                if in_docstring:
                    new_source.append(translate_docstring(line))
                    continue
                
                # Translate comments
                if stripped.startswith('#'):
                    new_source.append(translate_code_comment(line))
                elif '#' in line and not stripped.startswith('"') and not stripped.startswith("'"):
                    # Inline comment - only translate the comment part
                    # Find the # that's not inside a string
                    code_part = ''
                    comment_part = ''
                    in_string = False
                    string_char = None
                    for i, ch in enumerate(line):
                        if not in_string and ch in '"\'':
                            in_string = True
                            string_char = ch
                            code_part += ch
                        elif in_string and ch == string_char:
                            in_string = False
                            code_part += ch
                        elif not in_string and ch == '#':
                            comment_part = line[i:]
                            break
                        else:
                            code_part += ch
                    
                    if comment_part:
                        translated_comment = translate_code_comment(comment_part.strip())
                        new_source.append(code_part + translated_comment + '\n')
                    else:
                        new_source.append(line)
                else:
                    new_source.append(line)
            
            cell['source'] = new_source
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    
    print(f"Translated: {filepath}")


if __name__ == '__main__':
    import os
    base_dir = r"D:\repos\andrewNG\Course 1 - Supervised Machine Learning- Regression and Classification\Course 1 - Week 3\Practice Labs\archive"
    
    files = [
        "C1_W3_Logistic_Regression-Copy2.ipynb",
        "C1_W3_Logistic_Regression-Copy1.ipynb",
        "20230213_C1_W3_Logistic_Regression.ipynb",
    ]
    
    for f in files:
        filepath = os.path.join(base_dir, f)
        if os.path.exists(filepath):
            translate_notebook(filepath)
        else:
            print(f"File not found: {filepath}")
    
    print("Done!")
