// data.js
const ALGORITHMS = [


  // Linear Regression AlgoData
  {
    id: 'linear-regression',
    title: 'Linear Regression',
    category: 'supervised',
    difficulty: 'beginner',
    tags: ['Regression', 'Statistics', 'Supervised'],
    description: 'Models the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data.',
    icon: 'chart-line',
    lastUpdated: '2023-05-15',
    popularity: 0.95,
    
    // Core concept data
    concept: {
      overview: 'Linear regression is one of the most fundamental algorithms in machine learning. It assumes a linear relationship between input variables (X) and the single output variable (Y).',
      analogy: 'Imagine you\'re tracking how study time affects exam scores. Linear regression would help you find the relationship between these variables, allowing you to predict scores based on study hours.',
      history: 'First developed by Legendre and Gauss in the early 1800s for astronomical observations. Became fundamental in statistics and later machine learning.',
      mathematicalFormulation: {
        equation: 'y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε',
        variables: [
          { symbol: 'y', description: 'Dependent variable (target)' },
          { symbol: 'x₁...xₙ', description: 'Independent variables (features)' },
          { symbol: 'β₀', description: 'Y-intercept (bias term)' },
          { symbol: 'β₁...βₙ', description: 'Coefficients (weights)' },
          { symbol: 'ε', description: 'Error term (residuals)' }
        ],
        costFunction: 'MSE = (1/n) * Σ(yᵢ - ŷᵢ)²',
        optimization: 'Typically solved using Ordinary Least Squares (OLS) or Gradient Descent'
      },
      assumptions: [
        'Linear relationship between X and y',
        'Multivariate normality',
        'Little or no multicollinearity',
        'Homoscedasticity',
        'No auto-correlation'
      ]
    },
    
        // Enhanced visualization configuration
    visualization: {
      visualizerKey: 'linear-regression',
      defaultType: 'default',
      description: 'Interactive visualization of linear regression algorithm. Observe how the model finds the optimal line through data points using least squares optimization.',
      instructions: [
        'Adjust parameters to see real-time changes in the regression line',
        'Toggle residuals to visualize prediction errors',
        'Enable confidence intervals to see model uncertainty',
        'Change trend type (linear, quadratic, etc.) to explore different relationships'
      ],
      showTypeSelector: true,
      types: [
        { 
          value: 'default', 
          label: 'Standard View', 
          description: 'Basic scatter plot with regression line and residuals',
          default: true 
        },
        { 
          value: '2d-scatter', 
          label: '2D Scatter Plot',
          description: 'Focus on data distribution without regression line'
        },
        { 
          value: 'residual-plot', 
          label: 'Residual Analysis',
          description: 'Detailed view of prediction errors'
        },
        { 
          value: 'confidence-intervals', 
          label: 'Confidence Bands',
          description: 'Visualize model uncertainty'
        },
        { 
          value: 'all-intervals', 
          label: 'Complete View',
          description: 'Show all intervals and residuals together'
        }
      ],
      parameters: {
        n_samples: 100,
        noise: 0.5,
        slope: 2,
        intercept: 1,
        trend: 'linear',
        outliers: 0,
        show_residuals: true,
        show_confidence: false,
        show_prediction: false,
        show_stats: true,
        animation_duration: 1500,
        interactive: true
      },
      performanceTips: [
        'For smoother animations, reduce sample count below 200',
        'Confidence intervals require more computation - enable only when needed',
        'Outliers are highlighted in red when enabled'
      ]
    },
    
    implementations: {
      python: {
        version: "3.9",
        libraries: ['scikit-learn@1.0.2', 'statsmodels@0.13.2', 'numpy@1.21.5'],
        code: `# Python implementation using scikit-learn
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    import numpy as np

    # Generate sample data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create and fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    print(f"Intercept: {model.intercept_}")
    print(f"Coefficients: {model.coef_}")
    print(f"R² score: {model.score(X_test, y_test)}")`,
        timeComplexity: "O(n^2) for matrix inversion",
        spaceComplexity: "O(n^2)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
      "lastUpdated": "2025-08-29",
        notes: "This implementation uses scikit-learn for better numerical stability. For large datasets, consider using SGDRegressor instead."
      },
      r: {
        version: "4.2",
        libraries: ['stats@4.2.0', 'caret@6.0-93'],
        code: `# R implementation
    # Generate sample data
    set.seed(42)
    x <- runif(100, 0, 2)
    y <- 4 + 3 * x + rnorm(100)

    # Fit linear model
    model <- lm(y ~ x)

    # Print summary
    summary(model)

    # Plot results
    plot(x, y)
    abline(model, col = "red")`,
        timeComplexity: "O(n^2)",
        spaceComplexity: "O(n)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
      "lastUpdated": "2025-08-29",
        notes: "The base R implementation provides detailed statistical outputs. For larger datasets, consider using the 'biglm' package."
      },
      javascript: {
        version: "1.7.0",
        libraries: ['tensorflow.js@3.18.0', 'ml.js@0.12.0'],
        code: `// JavaScript implementation using TensorFlow.js
    import * as tf from '@tensorflow/tfjs';

    // Generate sample data
    const xs = tf.tensor1d([1, 2, 3, 4]);
    const ys = tf.tensor1d([2, 4, 6, 8]);

    // Define model
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));

    // Prepare for training
    model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

    // Train model
    await model.fit(xs, ys, {epochs: 250});

    // Predict
    model.predict(tf.tensor1d([5])).print();`,
        timeComplexity: "O(kn) per epoch",
        spaceComplexity: "O(n)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
      "lastUpdated": "2025-08-29",
        notes: "This implementation runs in the browser. For Node.js, use the '@tensorflow/tfjs-node' package for better performance."
      },
      cpp: {
        version: "17",
        libraries: ['Eigen@3.4.0'],
        code: `// C++ implementation using Eigen
    #include <Eigen/Dense>
    #include <iostream>

    using namespace Eigen;

    int main() {
      // Generate sample data
      MatrixXd X(100, 2);
      VectorXd y(100);
      
      for(int i = 0; i < 100; ++i) {
        X(i, 0) = 1.0;  // Intercept term
        X(i, 1) = 2.0 * rand() / RAND_MAX;
        y(i) = 4.0 + 3.0 * X(i, 1) + 0.1 * rand() / RAND_MAX;
      }

      // Solve linear regression
      VectorXd theta = X.bdcSvd(ComputeThinU | ComputeThinV).solve(y);
      
      // Output results
      std::cout << "Coefficients: " << theta.transpose() << std::endl;
      return 0;
    }`,
        timeComplexity: "O(n^3)",
        spaceComplexity: "O(n^2)",
        author: {
          name: "Numerical Computing Team"
        },
        lastUpdated: "2023-03-05",
        notes: "This implementation uses singular value decomposition for numerical stability. For very large matrices, consider using iterative methods."
      }
    },

    // Strengths and weaknesses
    prosCons: {
      strengths: [
        'Simple to implement and interpret',
        'Computationally efficient',
        'Performs well when relationship is linear',
        'Provides interpretable coefficients'
      ],
      weaknesses: [
        'Assumes linear relationship between variables',
        'Sensitive to outliers',
        'Assumes independence between features',
        'Tends to underfit with complex relationships'
      ]
    },
    
    hyperparameters: [
      {
        name: 'fit_intercept',
        type: 'boolean',
        default: true,
        description: 'Whether to calculate the intercept for this model'
      },
      {
        name: 'normalize',
        type: 'boolean',
        default: false,
        description: 'Whether to normalize the features before fitting'
      }
    ],
    
    // Use cases with dataset links
    useCases: [
      {
        title: 'Real Estate Pricing',
        description: 'Predicting house prices based on features like square footage, number of bedrooms, and location.',
        dataset: 'Boston Housing Prices',
        datasetLink: 'https://www.kaggle.com/datasets/vikrishnan/boston-house-prices'
      },
      {
        title: 'Sales Forecasting',
        description: 'Estimating future sales based on advertising spend across different channels.',
        dataset: 'Advertising Budget',
        datasetLink: 'https://www.kaggle.com/datasets/yasserh/advertising-sales-dataset'
      }
    ],

    
    // Comparison data
    comparisons: [
      {
        algorithm: 'Polynomial Regression',
        comparison: 'Linear regression cannot model nonlinear relationships, while polynomial regression can by adding polynomial terms'
      },
      {
        algorithm: 'Ridge/Lasso Regression',
        comparison: 'Linear regression can overfit with many features, while regularized versions prevent this'
      }
    ],
    
    quiz: [
      {
        question: 'What is the primary purpose of linear regression?',
        options: [
          'To classify data into categories',
          'To model relationships between variables',
          'To reduce dimensionality of data',
          'To cluster similar data points'
        ],
        correct: 1,
        explanation: 'Linear regression is used to model the linear relationship between independent and dependent variables.'
      },
      {
        question: 'Which of these is NOT an assumption of linear regression?',
        options: [
          'Linear relationship between X and y',
          'Features must be normally distributed',
          'No multicollinearity among features',
          'Homoscedasticity of residuals'
        ],
        correct: 1,
        explanation: 'While normality of residuals is assumed, the features themselves don\'t need to be normally distributed.'
      }
    ],
    
    // Mini projects
    projects: [
      {
        title: 'Predicting House Prices',
        description: 'Build a linear regression model to predict house prices based on features like square footage and number of bedrooms.',
        steps: [
          'Load and explore the Boston Housing dataset',
          'Preprocess the data (handle missing values, normalize features)',
          'Split into training and test sets',
          'Train a linear regression model',
          'Evaluate using R² score and RMSE',
          'Interpret the coefficients'
        ],
        difficulty: 'beginner',
        xp: 200
      }
    ]
  },
  
  // Logistic Regression AlgoData
  {
    id: 'logistic-regression',
    title: 'Logistic Regression',
    category: 'supervised',
    difficulty: 'beginner',
    tags: ['Classification', 'Probability', 'Supervised'],
    description: 'Estimates the probability of an event occurring by fitting data to a logistic curve. Widely used for binary classification problems.',
    icon: 'project-diagram',
    lastUpdated: '2023-06-22',
    popularity: 0.88,
    
    concept: {
      overview: 'Despite its name, logistic regression is a classification algorithm that models the probability of a binary outcome using a logistic function.',
      analogy: 'Imagine predicting whether an email is spam or not based on word frequencies. Logistic regression calculates the probability of it being spam.',
      history: 'Developed by statistician David Cox in 1958 as an extension of linear regression for binary outcomes.',
      mathematicalFormulation: {
        equation: 'P(y=1) = 1 / (1 + e^(-(β₀ + β₁x₁ + ... + βₙxₙ)))',
        variables: [
          { symbol: 'P(y=1)', description: 'Probability of positive class' },
          { symbol: 'e', description: 'Base of natural logarithm (~2.718)' },
          { symbol: 'β₀...βₙ', description: 'Model coefficients' }
        ],
        costFunction: 'Log Loss = -[y*log(p) + (1-y)*log(1-p)]',
        optimization: 'Typically solved using Maximum Likelihood Estimation (MLE)'
      },
      assumptions: [
        'Binary outcome variable',
        'Linear relationship between log-odds and features',
        'Little or no multicollinearity',
        'Large sample size (at least 10 events per predictor)'
      ]
    },
    
    visualization: {
      visualizerKey: 'logistic-regression',
      defaultType: 'decision-boundary',
      description: 'Interactive visualization of logistic regression classification. Observe how the algorithm creates non-linear decision boundaries.',
      instructions: [
        'Adjust class separation to change problem difficulty',
        'Modify learning rate to see training convergence speed',
        'Enable probability surface to see prediction confidence',
        'Toggle between different data distributions (linear, circular, XOR)'
      ],
      showTypeSelector: true,
      types: [
        { 
          value: 'decision-boundary', 
          label: 'Decision Boundary', 
          description: 'Shows how the algorithm separates classes',
          default: true 
        },
        { 
          value: 'probability-surface', 
          label: 'Probability Heatmap',
          description: 'Visualizes prediction probabilities'
        },
        { 
          value: 'decision-regions', 
          label: 'Decision Regions',
          description: 'Shows final classification areas'
        },
        { 
          value: 'with-loss', 
          label: 'Training Process',
          description: 'Includes loss curve during optimization'
        },
        { 
          value: 'all', 
          label: 'Complete View',
          description: 'Shows all visualization elements together'
        }
      ],
      parameters: {
        n_samples: 150,
        noise: 0.5,
        class_separation: 1.5,
        learning_rate: 0.1,
        regularization: 0.01,
        iterations: 100,
        distribution: 'linear',
        n_classes: 2,
        show_boundary: true,
        show_probability: false,
        show_decision: false,
        show_loss: false,
        animation_duration: 2000,
        interactive: true
      },
      performanceTips: [
        'Circular and XOR patterns show non-linear decision boundaries',
        'Probability surface rendering is more computationally intensive',
        'Higher learning rates may cause unstable training'
      ]
    },
    
    implementations: {
        "python": {
          "version": "3.9",
          "libraries": ["scikit-learn@1.0.2", "statsmodels@0.13.2", "numpy@1.21.5"],
          "code": `# Python implementation using scikit-learn
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    import numpy as np

    # Generate sample binary classification data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create and fit model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate
    print(f"Intercept: {model.intercept_}")
    print(f"Coefficients: {model.coef_}")
    print(f"Accuracy score: {model.score(X_test, y_test)}")`,
          "timeComplexity": "O(n^3) for small datasets, O(n) for large datasets with solver='sag'",
          "spaceComplexity": "O(n^2)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
      "lastUpdated": "2025-08-29",
          "notes": "For large datasets, consider using solver='sag' or 'saga'. For multiclass problems, set multi_class='multinomial'."
        },
        "r": {
          "version": "4.2",
          "libraries": ["stats@4.2.0", "caret@6.0-93"],
          "code": `# R implementation
    # Generate sample binary data
    set.seed(42)
    x1 <- rnorm(100)
    x2 <- rnorm(100)
    y <- as.factor(ifelse(x1 + x2 > 0, 1, 0))

    # Fit logistic regression model
    model <- glm(y ~ x1 + x2, family=binomial(link="logit"))

    # Print summary
    summary(model)

    # Predict probabilities
    pred_probs <- predict(model, type="response")`,
          "timeComplexity": "O(n^3)",
          "spaceComplexity": "O(n^2)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
      "lastUpdated": "2025-08-29",
          "notes": "For regularized logistic regression, use the glmnet package. For large datasets, consider speedglm."
        },
        "javascript": {
          "version": "1.7.0",
          "libraries": ["tensorflow.js@3.18.0", "ml.js@0.12.0"],
          "code": `// JavaScript implementation using TensorFlow.js
    import * as tf from '@tensorflow/tfjs';

    // Generate sample binary classification data
    const xs = tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]]);
    const ys = tf.tensor1d([0, 1, 1, 1]);

    // Define logistic regression model
    const model = tf.sequential();
    model.add(tf.layers.dense({
      units: 1,
      inputShape: [2],
      activation: 'sigmoid'
    }));

    // Prepare for training
    model.compile({
      optimizer: 'adam',
      loss: 'binaryCrossentropy',
      metrics: ['accuracy']
    });

    // Train model
    await model.fit(xs, ys, {
      epochs: 100,
      batchSize: 4
    });

    // Predict
    model.predict(tf.tensor2d([[0.5, 0.5]])).print();`,
          "timeComplexity": "O(kn) per epoch",
          "spaceComplexity": "O(n)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
      "lastUpdated": "2025-08-29",
          "notes": "For multiclass classification, change units to number of classes and use 'softmax' activation with 'categoricalCrossentropy' loss."
        },
        "cpp": {
          "version": "17",
          "libraries": ["Eigen@3.4.0", "dlib@19.24"],
          "code": `// C++ implementation using dlib
    #include <dlib/mlp.h>
    #include <iostream>

    using namespace dlib;

    int main() {
      // Create a simple logistic regression model (1-layer neural network)
      using net_type = loss_binary_log<fc<1, input<matrix<double>>>>;
      net_type net;
      
      // Generate sample data
      std::vector<matrix<double>> samples;
      std::vector<double> labels;
      
      // XOR-like data
      samples.push_back({0, 0}); labels.push_back(0);
      samples.push_back({0, 1}); labels.push_back(1);
      samples.push_back({1, 0}); labels.push_back(1);
      samples.push_back({1, 1}); labels.push_back(1);
      
      // Train the model
      dlib::sgd solver;
      solver.set_learning_rate(0.1);
      dlib::train_one_vs_one(solver, net, samples, labels, 1000);
      
      // Test prediction
      matrix<double> test_sample = {0.5, 0.5};
      double prediction = net(test_sample);
      std::cout << "Prediction: " << prediction << std::endl;
      
      return 0;
    }`,
          "timeComplexity": "O(kn) per iteration",
          "spaceComplexity": "O(n)",
          "author": {
            "name": "Numerical Computing Team"
          },
          "lastUpdated": "2023-03-05",
          "notes": "This implementation uses a simple neural network approach. For pure logistic regression without neural networks, consider implementing the gradient descent manually."
        }
      },
    prosCons: {
      strengths: [
        'Outputs probabilities with nice interpretation',
        'Fast to train and predict',
        'Works well with small datasets',
        'Less prone to overfitting with L1/L2 regularization'
      ],
      weaknesses: [
        'Assumes linear decision boundary',
        'Struggles with complex nonlinear relationships',
        'Sensitive to outliers',
        'Requires feature scaling for best performance'
      ]
    },
    
    hyperparameters: [
      {
        name: 'penalty',
        type: 'select',
        options: ['l1', 'l2', 'elasticnet', 'none'],
        default: 'l2',
        description: 'Type of regularization penalty to use'
      },
      {
        name: 'C',
        type: 'range',
        min: 0.01,
        max: 10,
        step: 0.01,
        default: 1.0,
        description: 'Inverse of regularization strength (smaller = stronger regularization)'
      }
    ],
    
    useCases: [
      {
        title: 'Credit Risk Assessment',
        description: 'Predicting whether a loan applicant is likely to default based on financial history.',
        dataset: 'German Credit Data'
      },
      {
        title: 'Disease Diagnosis',
        description: 'Classifying patients as having a disease or not based on medical test results.',
        dataset: 'Breast Cancer Wisconsin'
      }
    ],
    
    comparisons: [
      {
        algorithm: 'Linear Regression',
        comparison: 'Logistic regression is for classification while linear regression is for continuous outcomes'
      },
      {
        algorithm: 'SVM',
        comparison: 'Logistic regression provides probabilities while SVM provides hard classifications'
      }
    ],
    
    quiz: [
      {
        question: 'What is the output range of logistic regression?',
        options: [
          '0 to 1',
          '-∞ to +∞',
          'Any real number',
          '0 to 100'
        ],
        correct: 0,
        explanation: 'Logistic regression outputs probabilities between 0 and 1 through the sigmoid function.'
      }
    ],
    
    projects: [
      {
        title: 'Spam Classifier',
        description: 'Build a logistic regression model to classify emails as spam or not spam.',
        steps: [
          'Load and preprocess email text data',
          'Convert text to features using TF-IDF',
          'Train logistic regression model',
          'Evaluate using precision, recall, and ROC curve',
          'Interpret feature coefficients'
        ],
        difficulty: 'intermediate',
        xp: 300
      }
    ]
  },
  
  // Decision Tree AlgoData
  {
    id: 'decision-tree',
    title: 'Decision Tree',
    category: 'supervised',
    difficulty: 'beginner',
    tags: ['Classification', 'Regression', 'Supervised'],
    description: 'A tree-like model that makes decisions based on asking a series of questions about the input features.',
    icon: 'tree',
    lastUpdated: '2023-07-10',
    popularity: 0.85,
    
    concept: {
      overview: 'Decision trees recursively split the data based on feature values to maximize information gain at each step, creating a tree structure.',
      analogy: 'Like playing 20 Questions - each node in the tree asks a question that best divides the remaining possibilities.',
      history: 'Developed in the 1960s, with key algorithms like ID3 (1986), C4.5 (1993), and CART (1984).',
      mathematicalFormulation: {
        splittingCriteria: [
          {
            name: 'Gini Impurity',
            formula: 'G = 1 - Σ(pᵢ²)',
            description: 'Measure of node purity (0 = perfect purity)'
          },
          {
            name: 'Information Gain',
            formula: 'IG = H(parent) - Σ[(n_child/n_parent)*H(child)]',
            description: 'Reduction in entropy from split'
          }
        ],
        stoppingCriteria: [
          'Maximum depth reached',
          'Minimum samples per leaf',
          'No further information gain'
        ]
      }
    },
    
    visualization: {
      visualizerKey: 'decision-tree',
      defaultType: 'tree-split',
      description: 'Interactive visualization of decision tree algorithm. Watch the tree recursively partition the feature space.',
      instructions: [
        'Adjust tree depth to control model complexity',
        'Modify minimum samples per split to prevent overfitting',
        'Toggle between different data distributions',
        'Enable impurity metrics to understand split decisions'
      ],
      showTypeSelector: true,
      types: [
        { 
          value: 'tree-split', 
          label: 'Splitting Process', 
          description: 'Step-by-step tree building visualization',
          default: true 
        },
        { 
          value: 'decision-boundary', 
          label: 'Decision Regions',
          description: 'Final classification boundaries'
        },
        { 
          value: 'tree-diagram', 
          label: 'Tree Structure',
          description: 'Graphical representation of the tree'
        },
        { 
          value: 'all', 
          label: 'Complete View',
          description: 'All visualization elements together'
        },
        { 
          value: 'with-impurity', 
          label: 'With Metrics',
          description: 'Includes Gini impurity and samples info'
        }
      ],
      parameters: {
        n_samples: 200,
        n_classes: 3,
        max_depth: 3,
        min_samples_split: 2,
        min_samples_leaf: 1,
        distribution: 'concentric',
        show_tree: true,
        show_splits: true,
        show_regions: true,
        show_impurity: false,
        animation_duration: 2000,
        interactive: true
      },
      performanceTips: [
        'Trees deeper than 5 levels may be hard to visualize',
        'Checkerboard distribution shows axis-aligned splits clearly',
        'Impurity metrics help understand split decisions'
      ]
    },
    
      implementations: {
    "python": {
      "version": "3.9",
      "libraries": ["scikit-learn@1.0.2", "numpy@1.21.5", "pandas@1.3.5"],
      "code": `# Python implementation using scikit-learn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd

# Load sample data
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Evaluate
print(f"Training accuracy: {model.score(X_train, y_train):.3f}")
print(f"Test accuracy: {model.score(X_test, y_test):.3f}")

# Visualize (requires graphviz)
from sklearn.tree import export_graphviz
export_graphviz(model, out_file='tree.dot', 
                feature_names=data.feature_names,
                class_names=data.target_names,
                rounded=True, filled=True)`,
      "timeComplexity": "O(n²p) where n is samples and p is features",
      "spaceComplexity": "O(n)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
      "lastUpdated": "2025-08-29",
      "notes": "For regression problems, use DecisionTreeRegressor. Visualizing trees requires graphviz installed."
    },
    "r": {
      "version": "4.2",
      "libraries": ["rpart@4.1.16", "caret@6.0-93"],
      "code": `# R implementation using rpart
library(rpart)
library(rpart.plot)

# Load sample data
data(iris)

# Create decision tree model
model <- rpart(Species ~ ., data=iris, method="class", 
               control=rpart.control(maxdepth=3))

# Print summary
printcp(model)

# Plot tree
rpart.plot(model, type=4, extra=101)

# Predict
predictions <- predict(model, iris, type="class")
confusionMatrix(predictions, iris$Species)`,
      "timeComplexity": "O(n²p)",
      "spaceComplexity": "O(n)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
      "lastUpdated": "2025-08-29",
      "notes": "For regression trees, use method='anova'. The rpart.plot package provides better visualization options."
    },
    "javascript": {
      "version": "1.7.0",
      "libraries": ["ml.js@0.12.0"],
      "code": `// JavaScript implementation using ml.js
import { DecisionTreeClassifier } from 'ml-cart';

// Sample training data
const features = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
];
const labels = [0, 1, 1, 1];

// Create and train model
const options = {
  gainFunction: 'gini',
  maxDepth: 3,
  minNumSamples: 1
};
const classifier = new DecisionTreeClassifier(options);
classifier.train(features, labels);

// Predict
const test = [[0.5, 0.5]];
const prediction = classifier.predict(test);
console.log('Prediction:', prediction);

// Export tree as JSON
const treeModel = classifier.toJSON();
console.log('Model:', treeModel);`,
      "timeComplexity": "O(n²p)",
      "spaceComplexity": "O(n)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
      "lastUpdated": "2025-08-29",
      "notes": "For regression trees, use DecisionTreeRegression from the same package. The tree can be exported as JSON for visualization."
    },
    "cpp": {
      "version": "17",
      "libraries": ["mlpack@3.4.2"],
      "code": `// C++ implementation using mlpack
#include <mlpack.hpp>
#include <iostream>

using namespace mlpack;
using namespace mlpack::tree;
using namespace arma;

int main() {
  // Sample data (features and labels)
  mat features = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  Row<size_t> labels = {0, 1, 1, 1};

  // Initialize decision tree
  DecisionTree<> tree(features, labels, 2 /* numClasses */, 1 /* minLeafSize */);

  // Train model (mlpack trees are trained during construction)
  
  // Predict
  mat testPoint = {{0.5}, {0.5}};
  size_t prediction;
  tree.Classify(testPoint, prediction);
  std::cout << "Prediction: " << prediction << std::endl;

  return 0;
}`,
      "timeComplexity": "O(n²p)",
      "spaceComplexity": "O(n)",
      "author": {
        "name": "Numerical Computing Team"
      },
      "lastUpdated": "2023-03-05",
      "notes": "mlpack provides both classification and regression trees. For more control over tree parameters, use the DecisionTree constructor with additional parameters."
    }
  },
    
    prosCons: {
      strengths: [
        'Easy to understand and interpret',
        'Can handle both numerical and categorical data',
        'Requires little data preprocessing',
        'Non-parametric (no assumptions about data distribution)'
      ],
      weaknesses: [
        'Prone to overfitting without proper tuning',
        'Unstable (small changes can lead to completely different trees)',
        'Biased towards features with more levels',
        'Poor performance on linear relationships'
      ]
    },
    
    hyperparameters: [
      {
        name: 'max_depth',
        type: 'range',
        min: 1,
        max: 20,
        step: 1,
        default: 3,
        description: 'Maximum depth of the tree'
      },
      {
        name: 'min_samples_split',
        type: 'range',
        min: 2,
        max: 20,
        step: 1,
        default: 2,
        description: 'Minimum samples required to split an internal node'
      },
      {
        name: 'criterion',
        type: 'select',
        options: ['gini', 'entropy'],
        default: 'gini',
        description: 'Function to measure split quality'
      }
    ],
    
    useCases: [
      {
        title: 'Customer Churn Prediction',
        description: 'Predict whether customers will leave a service based on usage patterns.',
        dataset: 'Telco Customer Churn'
      }
    ],

    comparisons: [
      {
        algorithm: 'Random Forest',
        comparison: 'Random forests reduce overfitting by averaging multiple decision trees, while single trees can overfit.'
      },
      {
        algorithm: 'Logistic Regression',
        comparison: 'Decision trees can capture non-linear relationships, while logistic regression assumes linear decision boundaries.'
      }
    ],

    quiz: [
      {
        question: 'What is the primary advantage of decision trees?',
        options: [
          'They always achieve the highest accuracy',
          'They are easy to interpret and explain',
          'They require no parameter tuning',
          'They work best with linear relationships'
        ],
        correct: 1,
        explanation: 'Decision trees are highly interpretable as they mimic human decision-making processes with clear if-then rules.'
      },
      {
        question: 'Which metric is NOT typically used for splitting in decision trees?',
        options: [
          'Gini Impurity',
          'Information Gain',
          'Variance Reduction',
          'Euclidean Distance'
        ],
        correct: 3,
        explanation: 'Euclidean distance is used in clustering algorithms, not for splitting criteria in decision trees.'
      },
      {
        question: 'What is the main risk of using deep decision trees?',
        options: [
          'They become too simple',
          'They tend to underfit the data',
          'They are prone to overfitting',
          'They cannot handle categorical features'
        ],
        correct: 2,
        explanation: 'Deep decision trees can memorize noise in the training data, leading to overfitting and poor generalization.'
      },
      {
        question: 'How does a decision tree handle missing values?',
        options: [
          'It cannot handle missing values',
          'It uses surrogate splits',
          'It automatically imputes with mean values',
          'It removes instances with missing values'
        ],
        correct: 1,
        explanation: 'Many decision tree implementations use surrogate splits to handle missing values by finding similar splitting rules.'
      },
      {
        question: 'What is the purpose of pruning in decision trees?',
        options: [
          'To make the tree deeper',
          'To reduce overfitting by removing unnecessary branches',
          'To increase training speed',
          'To handle imbalanced datasets'
        ],
        correct: 1,
        explanation: 'Pruning removes branches that have little importance to reduce complexity and prevent overfitting.'
      }
    ],
    
    projects: [
      {
        title: 'Iris Species Classifier',
        description: 'Build a decision tree to classify iris flowers into three species.',
        steps: [
          'Load and explore the Iris dataset',
          'Visualize feature distributions',
          'Train decision tree with different parameters',
          'Visualize the decision boundaries',
          'Evaluate accuracy and interpret the tree'
        ],
        difficulty: 'beginner',
        xp: 250
      }
    ]
  },

    // Random Forest AlgoData
  {
    id: 'random-forest',
    title: 'Random Forest',
    category: 'ensemble',
    difficulty: 'beginner',
    tags: ['Ensemble', 'Classification', 'Regression', 'Supervised'],
    description: 'An ensemble learning method that constructs multiple decision trees and combines their predictions for improved accuracy and robustness.',
    icon: 'tree',
    lastUpdated: '2023-08-15',
    popularity: 0.92,
    
    concept: {
      overview: 'Random Forest builds multiple decision trees on random subsets of data and features, then aggregates their predictions to reduce overfitting and improve generalization.',
      analogy: 'Like asking multiple experts for their opinion on a problem and taking the majority vote - the collective wisdom is often better than any single expert.',
      history: 'Developed by Leo Breiman in 2001 as an extension of the bagging method, combining bootstrap aggregation with random feature selection.',
      mathematicalFormulation: {
        aggregationMethods: [
          {
            name: 'Majority Voting',
            formula: 'ŷ = mode({ŷ₁, ŷ₂, ..., ŷₙ})',
            description: 'For classification: select the most frequent prediction'
          },
          {
            name: 'Averaging',
            formula: 'ŷ = (1/n) * Σ(ŷᵢ)',
            description: 'For regression: average all tree predictions'
          }
        ],
        keyConcepts: [
          'Bootstrap sampling (bagging)',
          'Random feature selection',
          'Out-of-bag error estimation',
          'Feature importance scoring'
        ]
      },
      assumptions: [
        'Features should be relevant to prediction task',
        'No strict distributional assumptions',
        'Trees should be sufficiently diverse'
      ]
    },
    
    visualization: {
      visualizerKey: 'random-forest',
      defaultType: 'ensemble-view',
      description: 'Interactive visualization of random forest algorithm. Observe how multiple decision trees work together to create a more robust model.',
      instructions: [
        'Adjust number of trees to see ensemble effect',
        'Modify maximum depth to control tree complexity',
        'Toggle between individual tree views and ensemble prediction',
        'Observe how out-of-bag error changes with more trees'
      ],
      showTypeSelector: true,
      types: [
        { 
          value: 'ensemble-view', 
          label: 'Ensemble View', 
          description: 'Shows how multiple trees combine for final prediction',
          default: true 
        },
        { 
          value: 'individual-tree', 
          label: 'Individual Trees',
          description: 'Examine specific trees in the forest'
        },
        { 
          value: 'feature-importance', 
          label: 'Feature Importance',
          description: 'Visualize which features contribute most'
        },
        { 
          value: 'oob-error', 
          label: 'OOB Error',
          description: 'Track out-of-bag error during training'
        },
        { 
          value: 'all', 
          label: 'Complete View',
          description: 'All visualization elements together'
        }
      ],
      parameters: {
        n_trees: 10,
        max_depth: 4,
        n_samples: 200,
        n_classes: 3,
        bootstrap_ratio: 0.8,
        feature_subsampling: 0.7,
        show_individual_trees: false,
        show_ensemble: true,
        show_importance: false,
        show_oob: false,
        animation_duration: 2500,
        interactive: true
      },
      performanceTips: [
        'More trees generally improve performance but increase computation',
        'Feature importance helps understand model decisions',
        'OOB error provides unbiased performance estimate'
      ]
    },
    
    implementations: {
      python: {
        version: "3.9",
        libraries: ['scikit-learn@1.0.2', 'numpy@1.21.5'],
        code: `# Python implementation using scikit-learn
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import train_test_split
  from sklearn.datasets import load_iris
  import numpy as np

  # Load sample data
  data = load_iris()
  X, y = data.data, data.target

  # Split data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Create and fit model
  model = RandomForestClassifier(
      n_estimators=100,
      max_depth=3,
      random_state=42
  )
  model.fit(X_train, y_train)

  # Evaluate
  print(f"Training accuracy: {model.score(X_train, y_train):.3f}")
  print(f"Test accuracy: {model.score(X_test, y_test):.3f}")

  # Feature importance
  print("Feature importances:", model.feature_importances_)`,
        timeComplexity: "O(n_estimators * n²p) where n is samples and p is features",
        spaceComplexity: "O(n_estimators * n)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For regression problems, use RandomForestRegressor. Increase n_estimators for better performance but longer training time."
      },
      r: {
        version: "4.2",
        libraries: ['randomForest@4.7-1.1'],
        code: `# R implementation using randomForest package
  library(randomForest)

  # Load sample data
  data(iris)

  # Create random forest model
  model <- randomForest(Species ~ ., data=iris, 
                      ntree=100, mtry=2, importance=TRUE)

  # Print summary
  print(model)

  # Feature importance
  importance(model)
  varImpPlot(model)

  # Predict
  predictions <- predict(model, iris)
  table(predictions, iris$Species)`,
        timeComplexity: "O(ntree * n²p)",
        spaceComplexity: "O(ntree * n)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "The mtry parameter controls feature subsampling. Use importance=TRUE to compute feature importance scores."
      },
      javascript: {
        version: "1.7.0",
        libraries: ['ml.js@0.12.0'],
        code: `// JavaScript implementation using ml.js
  import { RandomForestClassifier } from 'ml-random-forest';

  // Sample training data
  const features = [
    [0, 0], [0, 1], [1, 0], [1, 1],
    [2, 2], [2, 3], [3, 2], [3, 3]
  ];
  const labels = [0, 0, 0, 0, 1, 1, 1, 1];

  // Create and train model
  const options = {
    seed: 42,
    maxFeatures: 1,
    replacement: true,
    nEstimators: 10
  };
  const classifier = new RandomForestClassifier(options);
  classifier.train(features, labels);

  // Predict
  const test = [[0.5, 0.5], [2.5, 2.5]];
  const predictions = classifier.predict(test);
  console.log('Predictions:', predictions);

  // Feature importance
  const importance = classifier.getFeatureImportance();
  console.log('Feature importance:', importance);`,
        timeComplexity: "O(n_estimators * n²p)",
        spaceComplexity: "O(n_estimators * n)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For regression, use RandomForestRegression. The seed parameter ensures reproducible results."
      },
      cpp: {
        version: "17",
        libraries: ['mlpack@3.4.2'],
        code: `// C++ implementation using mlpack
  #include <mlpack.hpp>
  #include <iostream>

  using namespace mlpack;
  using namespace arma;

  int main() {
    // Sample data
    mat features = randu<mat>(4, 100); // 4 features, 100 samples
    Row<size_t> labels = randi<Row<size_t>>(100, distr_param(0, 1)); // Binary labels

    // Initialize random forest
    RandomForest<> rf;
    
    // Train model
    rf.Train(features, labels, 2 /* numClasses */, 10 /* numTrees */);

    // Predict
    mat testPoint = randu<mat>(4, 1);
    size_t prediction;
    rf.Classify(testPoint, prediction);
    std::cout << "Prediction: " << prediction << std::endl;

    return 0;
  }`,
        timeComplexity: "O(numTrees * n²p)",
        spaceComplexity: "O(numTrees * n)",
        author: {
          name: "Numerical Computing Team"
        },
        lastUpdated: "2025-08-29",
        notes: "mlpack provides parallel training for random forests. For larger datasets, consider using the extraTrees parameter for Extremely Randomized Trees."
      }
    },

    prosCons: {
      strengths: [
        'Reduces overfitting compared to single decision trees',
        'Handles high dimensionality well',
        'Provides feature importance scores',
        'Works with both classification and regression'
      ],
      weaknesses: [
        'Less interpretable than single trees',
        'Can be computationally expensive',
        'May require more memory than other algorithms',
        'Can be slow for real-time predictions'
      ]
    },
    
    hyperparameters: [
      {
        name: 'n_estimators',
        type: 'range',
        min: 10,
        max: 500,
        step: 10,
        default: 100,
        description: 'Number of trees in the forest'
      },
      {
        name: 'max_depth',
        type: 'range',
        min: 1,
        max: 20,
        step: 1,
        default: 5,
        description: 'Maximum depth of each tree'
      },
      {
        name: 'max_features',
        type: 'select',
        options: ['auto', 'sqrt', 'log2', 0.5, 0.8],
        default: 'auto',
        description: 'Number of features to consider for best split'
      }
    ],
    
    useCases: [
      {
        title: 'Credit Risk Modeling',
        description: 'Predicting loan default risk based on financial and demographic features.',
        dataset: 'Lending Club Loan Data'
      },
      {
        title: 'Disease Prediction',
        description: 'Identifying patients at risk of developing chronic diseases.',
        dataset: 'Medical Health Records'
      }
    ],
    
    comparisons: [
      {
        algorithm: 'Decision Tree',
        comparison: 'Random Forest reduces overfitting and improves accuracy but is less interpretable'
      },
      {
        algorithm: 'Gradient Boosting',
        comparison: 'Random Forest builds trees independently while boosting builds them sequentially'
      }
    ],
    
    quiz: [
      {
        question: 'What is the primary technique used by Random Forest to create diverse trees?',
        options: [
          'Bagging and feature randomness',
          'Boosting and weight adjustment',
          'Stacking and meta-learning',
          'Bayesian averaging'
        ],
        correct: 0,
        explanation: 'Random Forest uses bootstrap aggregation (bagging) and random feature selection to create diverse trees.'
      },
      {
        question: 'How does Random Forest aggregate predictions from individual trees?',
        options: [
          'Majority voting for classification, averaging for regression',
          'Weighted averaging based on tree accuracy',
          'Neural network meta-learner',
          'Bayesian model averaging'
        ],
        correct: 0,
        explanation: 'For classification, it uses majority voting. For regression, it averages the predictions.'
      }
    ],
    
    projects: [
      {
        title: 'Predicting Customer Churn',
        description: 'Build a Random Forest model to predict which customers are likely to cancel their subscription.',
        steps: [
          'Load and explore customer data',
          'Preprocess features and handle missing values',
          'Train Random Forest with different parameters',
          'Evaluate using precision, recall, and ROC curve',
          'Analyze feature importance for business insights'
        ],
        difficulty: 'intermediate',
        xp: 350
      }
    ]
  },

  // K-Nearest Neighbors AlgoData
  {
    id: 'knn',
    title: 'K-Nearest Neighbors',
    category: 'supervised',
    difficulty: 'intermediate',
    tags: ['Classification', 'Regression', 'Instance-based', 'Supervised'],
    description: 'A non-parametric method that classifies or predicts based on the majority vote or average of the k closest training examples.',
    icon: 'users',
    lastUpdated: "2025-08-29",
    popularity: 0.82,
    
    concept: {
      overview: 'KNN is a simple, intuitive algorithm that stores all available cases and classifies new cases based on a similarity measure. It\'s a type of instance-based learning or lazy learning.',
      analogy: 'Imagine moving to a new neighborhood and wanting to know the dominant political affiliation. You\'d ask your k closest neighbors and go with the majority opinion.',
      history: 'First proposed by Evelyn Fix and Joseph Hodges in 1951 as a non-parametric classification method.',
      mathematicalFormulation: {
        distanceMetrics: [
          {
            name: 'Euclidean Distance',
            formula: 'd(x,y) = √Σ(xᵢ - yᵢ)²',
            description: 'Straight-line distance between points'
          },
          {
            name: 'Manhattan Distance',
            formula: 'd(x,y) = Σ|xᵢ - yᵢ|',
            description: 'Distance along axes at right angles'
          },
          {
            name: 'Minkowski Distance',
            formula: 'd(x,y) = (Σ|xᵢ - yᵢ|ᵖ)^(1/p)',
            description: 'Generalized distance metric'
          }
        ],
        votingMethods: [
          'Majority voting for classification',
          'Weighted voting (by distance)',
          'Averaging for regression'
        ]
      },
      assumptions: [
        'Feature space should be meaningful',
        'Data should be normalized',
        'Irrelevant features should be removed',
        'Appropriate distance metric should be chosen'
      ]
    },
    
    visualization: {
      visualizerKey: 'knn',
      defaultType: 'decision-boundary',
      description: 'Interactive visualization of KNN algorithm. Observe how the decision boundary changes with different k values and distance metrics.',
      instructions: [
        'Adjust k value to see how it affects decision boundaries',
        'Change distance metric (Euclidean, Manhattan, etc.)',
        'Toggle between weighted and unweighted voting',
        'Add new data points to see real-time classification'
      ],
      showTypeSelector: true,
      types: [
        { 
          value: 'decision-boundary', 
          label: 'Decision Boundary', 
          description: 'Shows how the algorithm classifies the feature space',
          default: true 
        },
        { 
          value: 'voronoi', 
          label: 'Voronoi Diagram',
          description: 'Visualizes regions of influence for each training point'
        },
        { 
          value: 'distance-metrics', 
          label: 'Distance Comparison',
          description: 'Compares different distance metrics'
        },
        { 
          value: 'k-comparison', 
          label: 'K Value Comparison',
          description: 'Shows decision boundaries for different k values'
        },
        { 
          value: 'all', 
          label: 'Complete View',
          description: 'All visualization elements together'
        }
      ],
      parameters: {
        n_samples: 100,
        n_classes: 3,
        k_value: 5,
        distance_metric: 'euclidean',
        weighted: false,
        show_boundary: true,
        show_voronoi: false,
        show_neighbors: true,
        highlight_neighbors: true,
        animation_duration: 1000,
        interactive: true
      },
      performanceTips: [
        'Smaller k values create more complex decision boundaries',
        'Manhattan distance works better with high-dimensional data',
        'Weighted voting gives more influence to closer neighbors'
      ]
    },
    
    implementations: {
      python: {
        version: "3.9",
        libraries: ['scikit-learn@1.0.2', 'numpy@1.21.5'],
        code: `# Python implementation using scikit-learn
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.model_selection import train_test_split
  from sklearn.datasets import load_iris
  from sklearn.preprocessing import StandardScaler

  # Load sample data
  data = load_iris()
  X, y = data.data, data.target

  # Preprocess data (important for KNN)
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  # Split data
  X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

  # Create and fit model
  model = KNeighborsClassifier(n_neighbors=5)
  model.fit(X_train, y_train)

  # Evaluate
  print(f"Training accuracy: {model.score(X_train, y_train):.3f}")
  print(f"Test accuracy: {model.score(X_test, y_test):.3f}")

  # Predict new sample
  new_sample = scaler.transform([[5.1, 3.5, 1.4, 0.2]])
  prediction = model.predict(new_sample)
  print(f"Predicted class: {data.target_names[prediction][0]}")`,
        timeComplexity: "O(1) for training, O(n) for prediction",
        spaceComplexity: "O(n)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For large datasets, consider using approximate nearest neighbors algorithms like Ball Tree or KD Tree for faster predictions."
      },
      r: {
        version: "4.2",
        libraries: ['class@7.3-20', 'caret@6.0-93'],
        code: `# R implementation using class package
  library(class)
  library(caret)

  # Load and preprocess data
  data(iris)
  set.seed(42)

  # Normalize data
  preproc <- preProcess(iris[,1:4], method=c("center", "scale"))
  iris_norm <- predict(preproc, iris[,1:4])

  # Split data
  train_index <- createDataPartition(iris$Species, p=0.8, list=FALSE)
  X_train <- iris_norm[train_index,]
  X_test <- iris_norm[-train_index,]
  y_train <- iris$Species[train_index]
  y_test <- iris$Species[-train_index]

  # Fit KNN model
  predictions <- knn(train=X_train, test=X_test, cl=y_train, k=5, prob=TRUE)

  # Evaluate
  confusionMatrix(predictions, y_test)`,
        timeComplexity: "O(1) for training, O(n) for prediction",
        spaceComplexity: "O(n)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For regression problems, use the knn.reg() function from the FNN package."
      },
      javascript: {
        version: "1.7.0",
        libraries: ['ml.js@0.12.0'],
        code: `// JavaScript implementation using ml.js
  import { KNN } from 'ml.js';

  // Sample training data
  const features = [
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],
    [5, 5],
    [5, 6],
    [6, 5],
    [6, 6]
  ];
  const labels = [0, 0, 0, 0, 1, 1, 1, 1];

  // Create and train model
  const options = {
    k: 3,
    distance: 'euclidean'
  };
  const classifier = new KNN(options);
  classifier.train(features, labels);

  // Predict
  const test = [[3, 3]];
  const prediction = classifier.predict(test);
  console.log('Prediction:', prediction);

  // Get probabilities
  const probabilities = classifier.predictProbability(test);
  console.log('Probabilities:', probabilities);`,
        timeComplexity: "O(1) for training, O(n) for prediction",
        spaceComplexity: "O(n)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For large datasets, consider implementing space partitioning structures like KD-trees for faster nearest neighbor searches."
      },
      cpp: {
        version: "17",
        libraries: ['mlpack@3.4.2'],
        code: `// C++ implementation using mlpack
  #include <mlpack.hpp>
  #include <iostream>

  using namespace mlpack;
  using namespace arma;

  int main() {
    // Sample data (features and labels)
    mat features = {{1,1}, {1,2}, {2,1}, {2,2}, {5,5}, {5,6}, {6,5}, {6,6}};
    Row<size_t> labels = {0, 0, 0, 0, 1, 1, 1, 1};

    // Normalize data (important for KNN)
    mat normalized_features;
    mlpack::data::Normalize(features, normalized_features);

    // Initialize KNN model
    KNN knn(normalized_features, labels, 1 /* neighborhood size */, true /* naive mode */);

    // Predict
    mat test_point = {{3, 3}};
    mat normalized_test;
    mlpack::data::Normalize(test_point, normalized_test);
    
    arma::Mat<size_t> predictions;
    arma::mat distances;
    knn.Search(normalized_test, 1, predictions, distances);
    
    std::cout << "Predicted class: " << predictions[0] << std::endl;
    std::cout << "Distance: " << distances[0] << std::endl;

    return 0;
  }`,
        timeComplexity: "O(1) for training, O(n) for prediction",
        spaceComplexity: "O(n)",
        author: {
          name: "Numerical Computing Team"
        },
        lastUpdated: "2025-08-29",
        notes: "For better performance with large datasets, use tree-based methods instead of naive search by setting the naive parameter to false."
      }
    },

    prosCons: {
      strengths: [
        'Simple to understand and implement',
        'No training phase (instance-based learning)',
        'Naturally handles multi-class problems',
        'Can be used for both classification and regression'
      ],
      weaknesses: [
        'Computationally expensive during prediction',
        'Sensitive to irrelevant features',
        'Performance depends on choice of k and distance metric',
        'Requires feature scaling'
      ]
    },
    
    hyperparameters: [
      {
        name: 'n_neighbors',
        type: 'range',
        min: 1,
        max: 20,
        step: 1,
        default: 5,
        description: 'Number of neighbors to use'
      },
      {
        name: 'weights',
        type: 'select',
        options: ['uniform', 'distance'],
        default: 'uniform',
        description: 'Weight function used in prediction'
      },
      {
        name: 'metric',
        type: 'select',
        options: ['euclidean', 'manhattan', 'minkowski'],
        default: 'euclidean',
        description: 'Distance metric to use'
      }
    ],
    
    useCases: [
      {
        title: 'Recommendation Systems',
        description: 'Finding similar users or items based on preferences or features.',
        dataset: 'MovieLens'
      },
      {
        title: 'Anomaly Detection',
        description: 'Identifying unusual patterns that are distant from normal instances.',
        dataset: 'Credit Card Fraud Detection'
      }
    ],
    
    comparisons: [
      {
        algorithm: 'Decision Trees',
        comparison: 'KNN makes no assumptions about data distribution while decision trees create axis-aligned splits'
      },
      {
        algorithm: 'SVM',
        comparison: 'KNN is instance-based and non-parametric while SVM finds a global decision boundary'
      }
    ],
    
    quiz: [
      {
        question: 'Why is KNN considered a "lazy" learning algorithm?',
        options: [
          'It doesn\'t require much computational power',
          'It postpones learning until prediction time',
          'It often performs poorly on complex tasks',
          'It requires minimal parameter tuning'
        ],
        correct: 1,
        explanation: 'KNN is called "lazy" because it doesn\'t learn a model during training but stores all training data and performs computation at prediction time.'
      },
      {
        question: 'What is the main computational bottleneck of KNN?',
        options: [
          'Training time',
          'Memory usage',
          'Prediction time',
          'Hyperparameter optimization'
        ],
        correct: 2,
        explanation: 'KNN prediction requires comparing the test instance to all training instances, which can be computationally expensive for large datasets.'
      }
    ],
    
    projects: [
      {
        title: 'Handwritten Digit Recognition',
        description: 'Build a KNN classifier to recognize handwritten digits from the MNIST dataset.',
        steps: [
          'Load and preprocess the MNIST dataset',
          'Normalize pixel values',
          'Train KNN with different k values',
          'Evaluate accuracy and find optimal k',
          'Visualize misclassified examples'
        ],
        difficulty: 'intermediate',
        xp: 350
      }
    ]
  },

  // Support Vector Machines AlgoData
  {
    id: 'svm',
    title: 'Support Vector Machines',
    category: 'supervised',
    difficulty: 'advanced',
    tags: ['Classification', 'Regression', 'Supervised', 'Kernel Method'],
    description: 'A powerful supervised learning algorithm that finds the optimal hyperplane that separates classes with the maximum margin.',
    icon: 'vector-square',
    lastUpdated: "2025-08-29",
    popularity: 0.87,
    
    concept: {
      overview: 'SVM finds the decision boundary that maximizes the margin between classes. It can handle both linear and non-linear classification using kernel tricks.',
      analogy: 'Imagine trying to separate two types of seeds on a table. SVM finds the widest possible "street" between the seeds, with the decision boundary right in the middle.',
      history: 'Developed by Vladimir Vapnik and colleagues in the 1960s-1990s. The kernel trick was introduced by Bernhard Boser, Isabelle Guyon, and Vapnik in 1992.',
      mathematicalFormulation: {
        optimizationProblem: 'min(1/2||w||² + CΣξᵢ) subject to yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0',
        keyConcepts: [
          {
            name: 'Support Vectors',
            description: 'Data points that define the margin boundaries'
          },
          {
            name: 'Margin',
            description: 'Distance between the decision boundary and the closest data points'
          },
          {
            name: 'Kernel Trick',
            description: 'Method to handle non-linear decision boundaries without explicitly transforming features'
          }
        ],
        commonKernels: [
          {
            name: 'Linear',
            formula: 'K(xᵢ, xⱼ) = xᵢᵀxⱼ'
          },
          {
            name: 'Polynomial',
            formula: 'K(xᵢ, xⱼ) = (γxᵢᵀxⱼ + r)ᵈ'
          },
          {
            name: 'RBF (Gaussian)',
            formula: 'K(xᵢ, xⱼ) = exp(-γ||xᵢ - xⱼ||²)'
          }
        ]
      },
      assumptions: [
        'Data should be standardized',
        'Works best when classes are separable',
        'Effective in high-dimensional spaces',
        'Kernel choice depends on data structure'
      ]
    },
    
    visualization: {
      visualizerKey: 'svm',
      defaultType: 'decision-boundary',
      description: 'Interactive visualization of SVM algorithm. Explore how different kernels and parameters affect the decision boundary.',
      instructions: [
        'Adjust C parameter to control margin hardness',
        'Change kernel type and parameters',
        'Toggle support vectors highlighting',
        'Modify gamma parameter for RBF kernel'
      ],
      showTypeSelector: true,
      types: [
        { 
          value: 'decision-boundary', 
          label: 'Decision Boundary', 
          description: 'Shows how the algorithm separates classes',
          default: true 
        },
        { 
          value: 'margin', 
          label: 'Margin Visualization',
          description: 'Highlights the margin and support vectors'
        },
        { 
          value: 'kernel-comparison', 
          label: 'Kernel Comparison',
          description: 'Compares different kernel functions'
        },
        { 
          value: 'parameter-effects', 
          label: 'Parameter Effects',
          description: 'Shows how C and gamma affect the boundary'
        },
        { 
          value: 'all', 
          label: 'Complete View',
          description: 'All visualization elements together'
        }
      ],
      parameters: {
        n_samples: 100,
        n_classes: 2,
        kernel: 'rbf',
        C: 1.0,
        gamma: 'scale',
        degree: 3,
        show_boundary: true,
        show_margin: true,
        show_support_vectors: true,
        show_contours: false,
        animation_duration: 2000,
        interactive: true
      },
      performanceTips: [
        'RBF kernel works well for most non-linear problems',
        'Small C creates softer margins, large C creates harder margins',
        'Gamma controls the influence of individual training examples'
      ]
    },
    
    implementations: {
      python: {
        version: "3.9",
        libraries: ['scikit-learn@1.0.2', 'numpy@1.21.5'],
        code: `# Python implementation using scikit-learn
  from sklearn.svm import SVC
  from sklearn.model_selection import train_test_split
  from sklearn.datasets import make_classification
  from sklearn.preprocessing import StandardScaler

  # Generate sample data
  X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                            n_informative=2, random_state=42, n_clusters_per_class=1)

  # Split data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Standardize data
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  # Create and fit model
  model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
  model.fit(X_train_scaled, y_train)

  # Evaluate
  print(f"Training accuracy: {model.score(X_train_scaled, y_train):.3f}")
  print(f"Test accuracy: {model.score(X_test_scaled, y_test):.3f}")
  print(f"Number of support vectors: {model.n_support_}")

  # Predict new sample
  new_sample = scaler.transform([[0.5, 0.5]])
  prediction = model.predict(new_sample)
  print(f"Predicted class: {prediction[0]}")`,
        timeComplexity: "O(n²) to O(n³) depending on kernel and parameters",
        spaceComplexity: "O(n²)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For large datasets, consider using LinearSVC which is more efficient. For very large problems, use SGDClassifier with hinge loss."
      },
      r: {
        version: "4.2",
        libraries: ['e1071@1.7-12', 'caret@6.0-93'],
        code: `# R implementation using e1071 package
  library(e1071)
  library(caret)

  # Generate sample data
  set.seed(42)
  data <- data.frame(
    x1 = rnorm(100),
    x2 = rnorm(100),
    y = as.factor(ifelse(x1^2 + x2^2 > 1, 1, 0))
  )

  # Split data
  train_index <- createDataPartition(data$y, p=0.8, list=FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]

  # Fit SVM model
  model <- svm(y ~ ., data=train_data, kernel="radial", cost=1, gamma=0.1)

  # Print summary
  summary(model)

  # Predict
  predictions <- predict(model, test_data)
  confusionMatrix(predictions, test_data$y)`,
        timeComplexity: "O(n²) to O(n³)",
        spaceComplexity: "O(n²)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "The e1071 package provides a comprehensive implementation of SVM. For large datasets, consider the LiblineaR package for linear SVMs."
      },
      javascript: {
        version: "1.7.0",
        libraries: ['ml.js@0.12.0'],
        code: `// JavaScript implementation using ml.js
  import { SVM } from 'ml.js';

  // Sample training data (XOR problem)
  const features = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
  ];
  const labels = [0, 1, 1, 0];

  // Create and train model
  const options = {
    kernel: 'rbf',
    C: 1.0,
    gamma: 0.1
  };
  const classifier = new SVM(options);
  classifier.train(features, labels);

  // Predict
  const test = [[0.5, 0.5]];
  const prediction = classifier.predict(test);
  console.log('Prediction:', prediction);

  // Get support vectors
  const supportVectors = classifier.getSupportVectors();
  console.log('Support vectors:', supportVectors);`,
        timeComplexity: "O(n²) to O(n³)",
        spaceComplexity: "O(n²)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "JavaScript implementations may be limited for large datasets. Consider using WebAssembly versions of LIBSVM for better performance."
      },
      cpp: {
        version: "17",
        libraries: ['mlpack@3.4.2'],
        code: `// C++ implementation using mlpack
  #include <mlpack.hpp>
  #include <iostream>

  using namespace mlpack;
  using namespace arma;

  int main() {
    // Sample data (XOR problem)
    mat features = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    Row<size_t> labels = {0, 1, 1, 0};

    // Initialize SVM model
    SVM<> svm(features, labels, 2 /* classes */, true /* fit intercept */);

    // Train model
    svm.Train(features, labels, 2);

    // Predict
    mat test_point = {{0.5, 0.5}};
    Row<size_t> prediction;
    svm.Classify(test_point, prediction);
    
    std::cout << "Predicted class: " << prediction[0] << std::endl;

    return 0;
  }`,
        timeComplexity: "O(n²) to O(n³)",
        spaceComplexity: "O(n²)",
        author: {
          name: "Numerical Computing Team"
        },
        lastUpdated: "2025-08-29",
        notes: "For non-linear problems, mlpack supports various kernels including linear, polynomial, and RBF. For large datasets, consider using the linear kernel for efficiency."
      }
    },

    prosCons: {
      strengths: [
        'Effective in high-dimensional spaces',
        'Memory efficient (uses only support vectors)',
        'Versatile through kernel functions',
        'Robust against overfitting in high-dimensional space'
      ],
      weaknesses: [
        'Does not perform well with large datasets',
        'Does not provide probability estimates directly',
        'Sensitive to kernel and parameter choices',
        'Can be difficult to interpret'
      ]
    },
    
    hyperparameters: [
      {
        name: 'C',
        type: 'range',
        min: 0.01,
        max: 100,
        step: 0.01,
        default: 1.0,
        description: 'Regularization parameter that controls trade-off between margin and classification error'
      },
      {
        name: 'kernel',
        type: 'select',
        options: ['linear', 'poly', 'rbf', 'sigmoid'],
        default: 'rbf',
        description: 'Kernel type to be used in the algorithm'
      },
      {
        name: 'gamma',
        type: 'range',
        min: 0.001,
        max: 10,
        step: 0.001,
        default: 0.1,
        description: 'Kernel coefficient for rbf, poly and sigmoid kernels'
      }
    ],
    
    useCases: [
      {
        title: 'Text Classification',
        description: 'Classifying documents into categories using high-dimensional text features.',
        dataset: '20 Newsgroups'
      },
      {
        title: 'Image Classification',
        description: 'Recognizing objects in images using features extracted from images.',
        dataset: 'CIFAR-10'
      }
    ],
    
    comparisons: [
      {
        algorithm: 'Logistic Regression',
        comparison: 'SVM finds the maximum margin decision boundary while logistic regression finds the probability-based decision boundary'
      },
      {
        algorithm: 'Neural Networks',
        comparison: 'SVM is based on convex optimization with global optimum, while neural networks can find complex boundaries but may get stuck in local minima'
      }
    ],
    
    quiz: [
      {
        question: 'What are support vectors?',
        options: [
          'The vectors that define the kernel function',
          'The data points closest to the decision boundary',
          'The eigenvectors of the covariance matrix',
          'The parameters that control the margin width'
        ],
        correct: 1,
        explanation: 'Support vectors are the data points that lie closest to the decision boundary and directly influence its position and orientation.'
      },
      {
        question: 'What is the kernel trick?',
        options: [
          'A method to speed up SVM training',
          'A way to handle missing data in SVM',
          'A technique to make linear algorithms work in non-linear spaces',
          'An approach to reduce the number of support vectors'
        ],
        correct: 2,
        explanation: 'The kernel trick allows SVM to operate in a high-dimensional feature space without explicitly computing the coordinates of the data in that space.'
      }
    ],
    
    projects: [
      {
        title: 'Handwritten Digit Recognition with SVM',
        description: 'Build an SVM classifier to recognize handwritten digits, comparing different kernels and parameters.',
        steps: [
          'Load and preprocess the MNIST dataset',
          'Extract relevant features',
          'Train SVM with different kernels (linear, RBF, polynomial)',
          'Tune hyperparameters using grid search',
          'Evaluate performance and analyze misclassifications'
        ],
        difficulty: 'intermediate',
        xp: 400
      }
    ]
  },

    // Quadratic Discriminant Analysis AlgoData
  {
    id: 'qda',
    title: 'Quadratic Discriminant Analysis',
    category: 'supervised',
    difficulty: 'advanced',
    tags: ['Classification', 'Statistical', 'Supervised'],
    description: 'A generative classification model that assumes each class has its own covariance matrix, allowing for quadratic decision boundaries.',
    icon: 'shapes',
    lastUpdated: '2023-07-28',
    popularity: 0.65,
    
    concept: {
      overview: 'QDA is a probabilistic classifier that models each class as a Gaussian distribution with its own covariance matrix, resulting in quadratic decision boundaries.',
      analogy: 'Like having different-shaped bubbles for each class - some are round, some are oval, and the boundaries between them are curved.',
      history: 'Developed as an extension of Linear Discriminant Analysis (LDA) to handle cases where classes have different covariance structures.',
      mathematicalFormulation: {
        discriminantFunction: 'δₖ(x) = -½log|Σₖ| - ½(x-μₖ)ᵀΣₖ⁻¹(x-μₖ) + logπₖ',
        decisionBoundary: 'Quadratic function of x',
        parameters: [
          { symbol: 'μₖ', description: 'Mean vector for class k' },
          { symbol: 'Σₖ', description: 'Covariance matrix for class k' },
          { symbol: 'πₖ', description: 'Prior probability of class k' }
        ]
      },
      assumptions: [
        'Each class follows a multivariate Gaussian distribution',
        'No assumption of equal covariance across classes',
        'Features should be continuous variables'
      ]
    },
    
    visualization: {
      visualizerKey: 'qda',
      defaultType: 'decision-boundary',
      description: 'Interactive visualization of QDA classification with quadratic decision boundaries.',
      instructions: [
        'Adjust class distributions to see different covariance structures',
        'Compare with LDA to see when quadratic boundaries are beneficial',
        'Toggle probability contours to see class distributions',
        'Observe how unequal covariances affect decision boundaries'
      ],
      showTypeSelector: true,
      types: [
        { 
          value: 'decision-boundary', 
          label: 'Decision Boundary', 
          description: 'Quadratic boundaries between classes',
          default: true 
        },
        { 
          value: 'probability-contours', 
          label: 'Probability Contours',
          description: 'Gaussian probability distributions for each class'
        },
        { 
          value: 'comparison-lda', 
          label: 'Comparison with LDA',
          description: 'Side-by-side comparison with linear discriminant analysis'
        },
        { 
          value: 'all', 
          label: 'Complete View',
          description: 'All visualization elements together'
        }
      ],
      parameters: {
        n_samples: 200,
        n_classes: 2,
        covariance_scale: 1.5,
        class_separation: 1.0,
        show_boundary: true,
        show_probability: false,
        show_comparison: false,
        animation_duration: 1500,
        interactive: true
      },
      performanceTips: [
        'QDA works best when classes have different covariance structures',
        'For high-dimensional data, QDA requires more parameters than LDA',
        'Visualizing probability contours is computationally intensive for many features'
      ]
    },
    
    implementations: {
      python: {
        version: "3.9",
        libraries: ['scikit-learn@1.0.2', 'numpy@1.21.5'],
        code: `# Python implementation using scikit-learn
  from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
  from sklearn.model_selection import train_test_split
  from sklearn.datasets import make_classification
  import numpy as np

  # Generate sample data with different covariances
  X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, 
                            n_redundant=0, n_clusters_per_class=1,
                            class_sep=1.0, random_state=42)

  # Split data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Create and fit QDA model
  model = QuadraticDiscriminantAnalysis()
  model.fit(X_train, y_train)

  # Evaluate
  print(f"Training accuracy: {model.score(X_train, y_train):.3f}")
  print(f"Test accuracy: {model.score(X_test, y_test):.3f}")

  # View model parameters
  print("Class means:", model.means_)
  print("Class priors:", model.priors_)`,
        timeComplexity: "O(n²p + p³) where p is number of features",
        spaceComplexity: "O(n × p + p²)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "QDA can be prone to overfitting with many features. Consider regularization or feature selection for high-dimensional data."
      },
      r: {
        version: "4.2",
        libraries: ['MASS@7.3-58.1'],
        code: `# R implementation using MASS
  library(MASS)

  # Generate sample data
  set.seed(42)
  n <- 1000
  X1 <- c(rnorm(n/2, mean = -1), rnorm(n/2, mean = 1))
  X2 <- c(rnorm(n/2, sd = 0.5), rnorm(n/2, sd = 1.5))
  X <- cbind(X1, X2)
  y <- factor(rep(0:1, each = n/2))

  # Fit QDA model
  model <- qda(X, y)

  # Predict
  predictions <- predict(model, X)
  table(Predicted = predictions$class, Actual = y)

  # View model summary
  print(model)`,
        timeComplexity: "O(n²p + p³)",
        spaceComplexity: "O(n × p + p²)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "The MASS package provides both LDA and QDA implementations. For high-dimensional data, consider regularized discriminant analysis."
      },
      javascript: {
        version: "1.7.0",
        libraries: ['ml.js@0.12.0'],
        code: `// JavaScript implementation using ml.js
  import { QDA } from 'ml-supervised';

  // Sample training data with different covariances
  const features = [
    [1.0, 1.2], [1.2, 0.8], [0.8, 1.0], // Class 0
    [3.0, 3.5], [3.2, 2.8], [2.8, 3.2]  // Class 1
  ];
  const labels = [0, 0, 0, 1, 1, 1];

  // Create and train QDA model
  const model = new QDA();
  model.train(features, labels);

  // Predict
  const test = [[1.5, 1.5], [2.5, 3.0]];
  const predictions = model.predict(test);
  console.log('Predictions:', predictions);

  // Get model parameters
  const summary = model.toJSON();
  console.log('Model summary:', summary);`,
        timeComplexity: "O(n²p + p³)",
        spaceComplexity: "O(n × p + p²)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "ml.js implementation is suitable for small to medium datasets. For large datasets, consider server-side processing."
      },
      cpp: {
        version: "17",
        libraries: ['mlpack@3.4.2', 'armadillo@11.2.1'],
        code: `// C++ implementation using mlpack
  #include <mlpack.hpp>
  #include <iostream>

  using namespace mlpack;
  using namespace arma;

  int main() {
    // Sample data (features and labels)
    mat features = randn<mat>(2, 100); // 2 features, 100 samples
    Row<size_t> labels = randi<Row<size_t>>(100, distr_param(0, 1)); // Binary labels
    
    // Create different covariance structure for each class
    for (size_t i = 0; i < 50; ++i) {
      features(1, i) *= 0.5; // Class 0 has smaller variance in second feature
    }
    for (size_t i = 50; i < 100; ++i) {
      features(0, i) *= 1.5; // Class 1 has larger variance in first feature
    }
    
    // Fit QDA model
    QuadraticDiscriminantAnalysis qda;
    qda.Train(features, labels);
    
    // Predict
    mat testPoint = { {1.0}, {1.0} };
    size_t prediction;
    qda.Classify(testPoint, prediction);
    std::cout << "Prediction: " << prediction << std::endl;
    
    return 0;
  }`,
        timeComplexity: "O(n²p + p³)",
        spaceComplexity: "O(n × p + p²)",
        author: {
          name: "Numerical Computing Team"
        },
        lastUpdated: "2025-08-29",
        notes: "mlpack provides efficient implementation of QDA. For very high-dimensional data, consider dimensionality reduction first."
      }
    },

    prosCons: {
      strengths: [
        'Can model more complex decision boundaries than LDA',
        'Works well when classes have different covariance structures',
        'Probabilistic output provides confidence estimates',
        'No hyperparameters to tune'
      ],
      weaknesses: [
        'Requires estimation of more parameters than LDA',
        'Can overfit with limited data or many features',
        'Assumes Gaussian distribution for each class',
        'Computationally more expensive than LDA'
      ]
    },
    
    hyperparameters: [
      {
        name: 'reg_param',
        type: 'range',
        min: 0.0,
        max: 1.0,
        step: 0.01,
        default: 0.0,
        description: 'Regularization parameter to prevent overfitting (0 = no regularization)'
      },
      {
        name: 'tol',
        type: 'range',
        min: 0.0001,
        max: 0.1,
        step: 0.0001,
        default: 0.0001,
        description: 'Threshold for singular value decomposition'
      }
    ],
    
    useCases: [
      {
        title: 'Medical Diagnosis',
        description: 'Classifying patients based on multiple clinical measurements with different variability patterns.',
        dataset: 'Biomedical Measurement Data'
      },
      {
        title: 'Quality Control',
        description: 'Identifying defective products based on multiple measurements with different distributions for good vs defective items.',
        dataset: 'Manufacturing Quality Data'
      }
    ],
    
    comparisons: [
      {
        algorithm: 'LDA',
        comparison: 'QDA allows different covariance matrices for each class while LDA assumes equal covariance matrices'
      },
      {
        algorithm: 'Logistic Regression',
        comparison: 'QDA makes distributional assumptions while logistic regression makes no such assumptions'
      }
    ],
    
    quiz: [
      {
        question: 'What is the key difference between QDA and LDA?',
        options: [
          'QDA uses quadratic features while LDA uses linear features',
          'QDA assumes different covariance matrices per class while LDA assumes equal covariance',
          'QDA is for regression while LDA is for classification',
          'QDA requires more data than LDA'
        ],
        correct: 1,
        explanation: 'The fundamental difference is that QDA allows each class to have its own covariance matrix, while LDA assumes all classes share the same covariance matrix.'
      },
      {
        question: 'When is QDA preferred over LDA?',
        options: [
          'When classes have similar covariance structures',
          'When classes have different covariance structures',
          'When there are many features',
          'When computational resources are limited'
        ],
        correct: 1,
        explanation: 'QDA is preferred when classes have different covariance structures, as it can model these differences with quadratic decision boundaries.'
      }
    ],
    
    projects: [
      {
        title: 'Iris Species Classification with QDA',
        description: 'Use QDA to classify iris flowers and compare with LDA performance.',
        steps: [
          'Load and explore the Iris dataset',
          'Visualize class distributions and covariance structures',
          'Train QDA and LDA models',
          'Compare decision boundaries and performance',
          'Analyze when QDA outperforms LDA'
        ],
        difficulty: 'intermediate',
        xp: 300
      }
    ]
  },

  // Linear Discriminant Analysis AlgoData
  {
    id: 'lda',
    title: 'Linear Discriminant Analysis',
    category: 'supervised',
    difficulty: 'intermediate',
    tags: ['Classification', 'Dimensionality Reduction', 'Supervised'],
    description: 'A statistical method that finds a linear combination of features that best separates two or more classes of objects or events.',
    icon: 'arrows-alt-h',
    lastUpdated: "2025-08-29",
    popularity: 0.78,
    
    concept: {
      overview: 'LDA finds the linear directions that maximize the separation between multiple classes while minimizing the variance within each class.',
      analogy: 'Like finding the best angle to view objects so that different types appear most separated from each other.',
      history: 'Developed by Ronald Fisher in 1936 as a method for classification and dimensionality reduction in statistics.',
      mathematicalFormulation: {
        objective: 'maximize (between-class variance) / (within-class variance)',
        discriminantFunction: 'δₖ(x) = xᵀΣ⁻¹μₖ - ½μₖᵀΣ⁻¹μₖ + logπₖ',
        parameters: [
          { symbol: 'μₖ', description: 'Mean vector for class k' },
          { symbol: 'Σ', description: 'Shared covariance matrix' },
          { symbol: 'πₖ', description: 'Prior probability of class k' }
        ]
      },
      assumptions: [
        'Each class follows a multivariate Gaussian distribution',
        'All classes share the same covariance matrix',
        'Features are continuous variables'
      ]
    },
    
    visualization: {
      visualizerKey: 'lda',
      defaultType: 'projection',
      description: 'Interactive visualization of LDA classification and dimensionality reduction.',
      instructions: [
        'Adjust class separation to see how it affects LDA performance',
        'Toggle between classification and dimensionality reduction views',
        'Observe how LDA finds the optimal projection direction',
        'Compare with PCA to see the difference between supervised and unsupervised methods'
      ],
      showTypeSelector: true,
      types: [
        { 
          value: 'projection', 
          label: 'Projection View', 
          description: 'Shows how LDA projects data to maximize separation',
          default: true 
        },
        { 
          value: 'decision-boundary', 
          label: 'Decision Boundary',
          description: 'Linear boundaries between classes'
        },
        { 
          value: 'comparison-pca', 
          label: 'Comparison with PCA',
          description: 'Side-by-side comparison with principal component analysis'
        },
        { 
          value: 'all', 
          label: 'Complete View',
          description: 'All visualization elements together'
        }
      ],
      parameters: {
        n_samples: 200,
        n_classes: 3,
        n_features: 2,
        class_separation: 1.5,
        show_projection: true,
        show_boundary: false,
        show_comparison: false,
        animation_duration: 1500,
        interactive: true
      },
      performanceTips: [
        'LDA works best when the equal covariance assumption holds',
        'For visualization, start with 2-3 classes and 2 features',
        'Comparing with PCA helps understand the difference between supervised and unsupervised methods'
      ]
    },
    
    implementations: {
      python: {
        version: "3.9",
        libraries: ['scikit-learn@1.0.2', 'numpy@1.21.5'],
        code: `# Python implementation using scikit-learn
  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
  from sklearn.model_selection import train_test_split
  from sklearn.datasets import load_iris
  import numpy as np

  # Load sample data
  data = load_iris()
  X, y = data.data, data.target

  # Split data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Create and fit LDA model
  model = LinearDiscriminantAnalysis()
  model.fit(X_train, y_train)

  # Evaluate
  print(f"Training accuracy: {model.score(X_train, y_train):.3f}")
  print(f"Test accuracy: {model.score(X_test, y_test):.3f}")

  # Use for dimensionality reduction
  X_lda = model.transform(X_test)
  print(f"Original shape: {X_test.shape}, LDA shape: {X_lda.shape}")`,
        timeComplexity: "O(n × p² + p³) where p is number of features",
        spaceComplexity: "O(n × p + p²)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "LDA can be used for both classification and dimensionality reduction. For dimensionality reduction, it finds at most (number of classes - 1) components."
      },
      r: {
        version: "4.2",
        libraries: ['MASS@7.3-58.1'],
        code: `# R implementation using MASS
  library(MASS)

  # Load sample data
  data(iris)

  # Fit LDA model
  model <- lda(Species ~ ., data=iris)

  # Print model summary
  print(model)

  # Predict
  predictions <- predict(model, iris)
  table(Predicted = predictions$class, Actual = iris$Species)

  # View discriminant functions
  plot(model, dimen=2)`,
        timeComplexity: "O(n × p² + p³)",
        spaceComplexity: "O(n × p + p²)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "The MASS package provides comprehensive LDA implementation. Use the plot() function to visualize the discriminant functions."
      },
      javascript: {
        version: "1.7.0",
        libraries: ['ml.js@0.12.0'],
        code: `// JavaScript implementation using ml.js
  import { LDA } from 'ml-supervised';

  // Sample training data
  const features = [
    [1.0, 1.2], [1.2, 0.8], [0.8, 1.0], // Class 0
    [2.0, 2.2], [2.2, 1.8], [1.8, 2.0], // Class 1
    [3.0, 3.2], [3.2, 2.8], [2.8, 3.0]  // Class 2
  ];
  const labels = [0, 0, 0, 1, 1, 1, 2, 2, 2];

  // Create and train LDA model
  const model = new LDA();
  model.train(features, labels);

  // Predict
  const test = [[1.5, 1.5], [2.5, 2.5]];
  const predictions = model.predict(test);
  console.log('Predictions:', predictions);

  // Get model parameters
  const summary = model.toJSON();
  console.log('Model summary:', summary);`,
        timeComplexity: "O(n × p² + p³)",
        spaceComplexity: "O(n × p + p²)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "ml.js implementation is suitable for small to medium datasets. For dimensionality reduction, the transform method projects data to the discriminant space."
      },
      cpp: {
        version: "17",
        libraries: ['mlpack@3.4.2', 'armadillo@11.2.1'],
        code: `// C++ implementation using mlpack
  #include <mlpack.hpp>
  #include <iostream>

  using namespace mlpack;
  using namespace arma;

  int main() {
    // Sample data (features and labels)
    mat features = randn<mat>(4, 150); // 4 features, 150 samples
    Row<size_t> labels = randi<Row<size_t>>(150, distr_param(0, 2)); // 3 classes
    
    // Create different means for each class
    for (size_t i = 0; i < 50; ++i) {
      features.col(i) += vec{0, 0, 0, 0}; // Class 0 mean
    }
    for (size_t i = 50; i < 100; ++i) {
      features.col(i) += vec{2, 2, 2, 2}; // Class 1 mean
    }
    for (size_t i = 100; i < 150; ++i) {
      features.col(i) += vec{4, 4, 4, 4}; // Class 2 mean
    }
    
    // Fit LDA model
    LDA lda;
    lda.Train(features, labels, 3); // 3 classes
    
    // Predict
    mat testPoint = { {1.0}, {1.0}, {1.0}, {1.0} };
    size_t prediction;
    lda.Classify(testPoint, prediction);
    std::cout << "Prediction: " << prediction << std::endl;
    
    return 0;
  }`,
        timeComplexity: "O(n × p² + p³)",
        spaceComplexity: "O(n × p + p²)",
        author: {
          name: "Numerical Computing Team"
        },
        lastUpdated: "2025-08-29",
        notes: "mlpack provides efficient implementation of LDA. For dimensionality reduction, use the Transform method to project data to the discriminant space."
      }
    },

    prosCons: {
      strengths: [
        'Simple and computationally efficient',
        'Provides probabilistic classification',
        'Can be used for dimensionality reduction',
        'Works well when the equal covariance assumption holds'
      ],
      weaknesses: [
        'Assumes Gaussian distribution for each class',
        'Assumes equal covariance matrices across classes',
        'Limited to linear decision boundaries',
        'Performance degrades when assumptions are violated'
      ]
    },
    
    hyperparameters: [
      {
        name: 'solver',
        type: 'select',
        options: ['svd', 'lsqr', 'eigen'],
        default: 'svd',
        description: 'Solver to use for computing the discriminant functions'
      },
      {
        name: 'shrinkage',
        type: 'range',
        min: 0.0,
        max: 1.0,
        step: 0.01,
        default: 0.0,
        description: 'Shrinkage parameter for regularization (useful when n_features > n_samples)'
      }
    ],
    
    useCases: [
      {
        title: 'Face Recognition',
        description: 'Dimensionality reduction and classification for facial recognition systems.',
        dataset: 'ORL, Yale Face Database'
      },
      {
        title: 'Customer Segmentation',
        description: 'Identifying distinct customer groups based on purchasing behavior and demographics.',
        dataset: 'Retail Customer Data'
      }
    ],
    
    comparisons: [
      {
        algorithm: 'Logistic Regression',
        comparison: 'LDA makes distributional assumptions while logistic regression makes no such assumptions'
      },
      {
        algorithm: 'QDA',
        comparison: 'LDA assumes equal covariance matrices while QDA allows different covariance matrices per class'
      }
    ],
    
    quiz: [
      {
        question: 'What is the main objective of Linear Discriminant Analysis?',
        options: [
          'To minimize within-class variance',
          'To maximize between-class variance',
          'To maximize the ratio of between-class to within-class variance',
          'To find the principal components of the data'
        ],
        correct: 2,
        explanation: 'LDA aims to find the linear combinations that maximize the ratio of between-class variance to within-class variance.'
      },
      {
        question: 'How many discriminant functions can LDA find for a classification problem with K classes?',
        options: [
          'K functions',
          'K-1 functions',
          'As many as the number of features',
          'Exactly 2 functions regardless of K'
        ],
        correct: 1,
        explanation: 'LDA can find at most K-1 discriminant functions, where K is the number of classes.'
      }
    ],
    
    projects: [
      {
        title: 'Wine Classification with LDA',
        description: 'Use LDA to classify wines and reduce dimensionality for visualization.',
        steps: [
          'Load the Wine dataset',
          'Perform exploratory data analysis',
          'Apply LDA for classification',
          'Use LDA for dimensionality reduction to 2D',
          'Visualize the results and decision boundaries'
        ],
        difficulty: 'intermediate',
        xp: 350
      }
    ]
  },

    // Naive Bayes AlgoData
  {
    id: 'naive-bayes',
    title: 'Naive Bayes',
    category: 'supervised',
    difficulty: 'intermediate',
    tags: ['Classification', 'Probability', 'Supervised'],
    description: 'A family of simple probabilistic classifiers based on applying Bayes theorem with strong independence assumptions between features.',
    icon: 'chart-pie',
    lastUpdated: '2025-08-29',
    popularity: 0.78,
    
    concept: {
      overview: 'Naive Bayes classifiers apply Bayes theorem with the "naive" assumption of conditional independence between every pair of features given the class value.',
      analogy: 'Like a doctor who assumes symptoms are independent when making a diagnosis - fever and cough might be related, but treating them as independent simplifies the calculation.',
      history: 'Rooted in 18th century Bayesian probability theory, with the "naive" independence assumption formalized in the 1960s for text classification.',
      mathematicalFormulation: {
        theorem: 'P(y|X) = P(X|y) * P(y) / P(X)',
        naiveAssumption: 'P(X|y) = Π P(xᵢ|y) for i = 1 to n features',
        keyComponents: [
          {
            symbol: 'P(y)',
            description: 'Prior probability of class y'
          },
          {
            symbol: 'P(X|y)',
            description: 'Likelihood of features X given class y'
          },
          {
            symbol: 'P(X)',
            description: 'Evidence (marginal probability of features)'
          },
          {
            symbol: 'P(y|X)',
            description: 'Posterior probability of class y given features X'
          }
        ]
      },
      variants: [
        'Gaussian Naive Bayes: For continuous features assuming normal distribution',
        'Multinomial Naive Bayes: For discrete count data (e.g., text classification)',
        'Bernoulli Naive Bayes: For binary/boolean features'
      ]
    },
    
    visualization: {
      visualizerKey: 'naive-bayes',
      defaultType: 'probability-surface',
      description: 'Interactive visualization of Naive Bayes classification. Observe how probability distributions influence decision boundaries.',
      instructions: [
        'Adjust feature distributions to see different decision boundaries',
        'Toggle between different Naive Bayes variants',
        'Observe how prior probabilities affect classification',
        'Enable probability contours to see confidence levels'
      ],
      showTypeSelector: true,
      types: [
        { 
          value: 'probability-surface', 
          label: 'Probability Surface', 
          description: 'Shows probability distributions and decision boundaries',
          default: true 
        },
        { 
          value: 'feature-distributions', 
          label: 'Feature Distributions',
          description: 'Visualizes the assumed distribution of each feature'
        },
        { 
          value: 'decision-boundary', 
          label: 'Decision Boundary',
          description: 'Shows the final classification regions'
        },
        { 
          value: 'all', 
          label: 'Complete View',
          description: 'All visualization elements together'
        },
        { 
          value: 'with-priors', 
          label: 'With Priors',
          description: 'Includes prior probability adjustments'
        }
      ],
      parameters: {
        n_samples: 150,
        n_classes: 2,
        distribution_type: 'gaussian',
        class_separation: 1.0,
        show_distributions: true,
        show_boundary: true,
        show_probability: false,
        show_priors: false,
        animation_duration: 1500,
        interactive: true
      },
      performanceTips: [
        'Gaussian variant works best with normally distributed features',
        'Multinomial variant is ideal for text classification',
        'Decision boundaries are linear for Gaussian Naive Bayes'
      ]
    },
    
    implementations: {
      python: {
        version: "3.9",
        libraries: ['scikit-learn@1.0.2', 'numpy@1.21.5'],
        code: `# Python implementation using scikit-learn
  from sklearn.naive_bayes import GaussianNB
  from sklearn.model_selection import train_test_split
  from sklearn.datasets import load_iris
  from sklearn.metrics import accuracy_score

  # Load sample data
  data = load_iris()
  X, y = data.data, data.target

  # Split data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Create and fit model
  model = GaussianNB()
  model.fit(X_train, y_train)

  # Predict
  y_pred = model.predict(X_test)
  print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

  # View parameters
  print("Class priors:", model.class_prior_)
  print("Class counts:", model.class_count_)
  print("Theta (mean):", model.theta_)
  print("Sigma (variance):", model.sigma_)`,
        timeComplexity: "O(n * p) where n is samples and p is features",
        spaceComplexity: "O(c * p) where c is classes and p is features",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For text classification, use MultinomialNB. For binary features, use BernoulliNB. GaussianNB assumes features follow normal distribution."
      },
      r: {
        version: "4.2",
        libraries: ['e1071@1.7-12'],
        code: `# R implementation using e1071 package
  library(e1071)

  # Load sample data
  data(iris)

  # Create Naive Bayes model
  model <- naiveBayes(Species ~ ., data=iris)

  # Print summary
  print(model)

  # Predict
  predictions <- predict(model, iris)
  table(predictions, iris$Species)

  # Get prediction probabilities
  probabilities <- predict(model, iris, type="raw")
  head(probabilities)`,
        timeComplexity: "O(n * p)",
        spaceComplexity: "O(c * p)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "The e1071 package provides various Naive Bayes implementations. For more control, use the klaR package."
      },
      javascript: {
        version: "1.7.0",
        libraries: ['ml.js@0.12.0'],
        code: `// JavaScript implementation using ml.js
  import { GaussianNB } from 'ml-naivebayes';

  // Sample training data
  const features = [
    [0, 0], [0, 1], [1, 0], [1, 1],
    [2, 2], [2, 3], [3, 2], [3, 3]
  ];
  const labels = [0, 0, 0, 0, 1, 1, 1, 1];

  // Create and train model
  const classifier = new GaussianNB();
  classifier.train(features, labels);

  // Predict
  const test = [[0.5, 0.5], [2.5, 2.5]];
  const predictions = classifier.predict(test);
  console.log('Predictions:', predictions);

  // Get prediction probabilities
  const probabilities = classifier.predictProba(test);
  console.log('Probabilities:', probabilities);`,
        timeComplexity: "O(n * p)",
        spaceComplexity: "O(c * p)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "ml.js also provides MultinomialNB and BernoulliNB implementations. GaussianNB is the default for continuous features."
      },
      cpp: {
        version: "17",
        libraries: ['mlpack@3.4.2'],
        code: `// C++ implementation using mlpack
  #include <mlpack.hpp>
  #include <iostream>

  using namespace mlpack;
  using namespace arma;

  int main() {
    // Sample data
    mat features = randn<mat>(4, 100); // 4 features, 100 samples
    Row<size_t> labels = randi<Row<size_t>>(100, distr_param(0, 1)); // Binary labels

    // Create and train Naive Bayes classifier
    NaiveBayesClassifier<> nb;
    nb.Train(features, labels, 2 /* numClasses */);

    // Predict
    mat testPoint = randn<mat>(4, 1);
    size_t prediction;
    nb.Classify(testPoint, prediction);
    std::cout << "Prediction: " << prediction << std::endl;

    return 0;
  }`,
        timeComplexity: "O(n * p)",
        spaceComplexity: "O(c * p)",
        author: {
          name: "Numerical Computing Team"
        },
        lastUpdated: "2025-08-29",
        notes: "mlpack assumes Gaussian distribution for all features. For discrete features, preprocess them appropriately."
      }
    },

    prosCons: {
      strengths: [
        'Extremely fast training and prediction',
        'Works well with high-dimensional data',
        'Requires small amount of training data',
        'Handles both continuous and discrete data'
      ],
      weaknesses: [
        'Strong independence assumption rarely holds in practice',
        'Can be outperformed by more complex models',
        'Probability estimates can be inaccurate',
        'Sensitive to feature representation'
      ]
    },
    
    hyperparameters: [
      {
        name: 'var_smoothing',
        type: 'range',
        min: 1e-9,
        max: 1e-1,
        step: 1e-9,
        default: 1e-9,
        description: 'Portion of the largest variance of all features added to variances for calculation stability'
      },
      {
        name: 'priors',
        type: 'select',
        options: ['None', 'Balanced', 'Custom'],
        default: 'None',
        description: 'Prior probabilities of the classes. If specified, priors are not adjusted according to the data'
      }
    ],
    
    useCases: [
      {
        title: 'Spam Detection',
        description: 'Classifying emails as spam or not spam based on word frequencies.',
        dataset: 'Enron Spam Dataset'
      },
      {
        title: 'Sentiment Analysis',
        description: 'Classifying text as positive, negative, or neutral sentiment.',
        dataset: 'IMDB Movie Reviews'
      },
      {
        title: 'Medical Diagnosis',
        description: 'Predicting diseases based on symptoms and test results.',
        dataset: 'UCI Heart Disease Dataset'
      }
    ],
    
    comparisons: [
      {
        algorithm: 'Logistic Regression',
        comparison: 'Both are linear classifiers but Naive Bayes makes stronger independence assumptions'
      },
      {
        algorithm: 'Support Vector Machines',
        comparison: 'Naive Bayes is faster and probabilistic but SVMs often achieve higher accuracy'
      }
    ],
    
    quiz: [
      {
        question: 'What is the "naive" assumption in Naive Bayes classifiers?',
        options: [
          'Features are conditionally independent given the class',
          'All features are equally important',
          'Classes are balanced in the dataset',
          'Features follow normal distribution'
        ],
        correct: 0,
        explanation: 'The "naive" assumption is that all features are conditionally independent given the class value, which simplifies the probability calculations.'
      },
      {
        question: 'In which application is Naive Bayes particularly effective despite its simplicity?',
        options: [
          'Text classification',
          'Image recognition',
          'Time series forecasting',
          'Reinforcement learning'
        ],
        correct: 0,
        explanation: 'Naive Bayes is particularly effective for text classification tasks like spam detection and sentiment analysis, where the "bag of words" representation aligns well with the independence assumption.'
      }
    ],
    
    projects: [
      {
        title: 'Email Spam Classifier',
        description: 'Build a Naive Bayes classifier to detect spam emails.',
        steps: [
          'Load and preprocess email text data',
          'Convert text to feature vectors using TF-IDF',
          'Train Multinomial Naive Bayes model',
          'Evaluate using precision, recall, and F1-score',
          'Deploy as a simple web application'
        ],
        difficulty: 'beginner',
        xp: 200
      }
    ]
  },

  // K-Means Clustering AlgoData
  {
    id: 'k-means',
    title: 'K-Means Clustering',
    category: 'unsupervised',
    difficulty: 'beginner',
    tags: ['Clustering', 'Unsupervised', 'Partitioning'],
    description: 'Partitions data into K distinct clusters based on feature similarity, where each observation belongs to the cluster with the nearest mean.',
    icon: 'object-group',
    lastUpdated: '2025-08-29',
    popularity: 0.82,
    
    concept: {
      overview: 'K-means clustering aims to partition n observations into k clusters where each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster.',
      analogy: 'Imagine organizing a room full of people into groups where people in the same group are most similar to each other in terms of height, weight, and age.',
      history: 'First proposed by Stuart Lloyd in 1957 as a pulse-code modulation technique, later published in 1982. The standard algorithm was independently developed by Forgy in 1965.',
      mathematicalFormulation: {
        objectiveFunction: 'J = ΣΣ||xᵢ - μⱼ||²',
        variables: [
          { symbol: 'K', description: 'Number of clusters' },
          { symbol: 'μⱼ', description: 'Centroid of cluster j' },
          { symbol: 'xᵢ', description: 'Data point i' },
          { symbol: 'cᵢ', description: 'Cluster assignment for point i' }
        ],
        algorithmSteps: [
          'Initialize K cluster centroids randomly',
          'Assign each point to the nearest centroid',
          'Recalculate centroids as mean of assigned points',
          'Repeat until assignments stop changing'
        ]
      },
      assumptions: [
        'Clusters are spherical and equally sized',
        'Clusters are well-separated',
        'The value of K is known or can be reasonably estimated'
      ]
    },
    
    visualization: {
      visualizerKey: 'k-means',
      defaultType: 'cluster-formation',
      description: 'Interactive visualization of k-means clustering algorithm. Observe how centroids move and data points are assigned to clusters.',
      instructions: [
        'Adjust the number of clusters (K) to see different groupings',
        'Watch centroids move toward the center of their clusters',
        'Toggle between different initialization methods',
        'Enable/disable animation to see the iterative process'
      ],
      showTypeSelector: true,
      types: [
        { 
          value: 'cluster-formation', 
          label: 'Cluster Formation', 
          description: 'Step-by-step visualization of clustering process',
          default: true 
        },
        { 
          value: 'centroid-movement', 
          label: 'Centroid Movement',
          description: 'Focus on how centroids evolve during iterations'
        },
        { 
          value: 'voronoi-diagram', 
          label: 'Voronoi Diagram',
          description: 'Shows decision boundaries between clusters'
        },
        { 
          value: 'elbow-method', 
          label: 'Elbow Method',
          description: 'Helps determine optimal number of clusters'
        },
        { 
          value: 'all', 
          label: 'Complete View',
          description: 'All visualization elements together'
        }
      ],
      parameters: {
        n_samples: 300,
        n_clusters: 3,
        cluster_std: 0.8,
        init_method: 'k-means++',
        max_iter: 300,
        random_state: 42,
        show_centroids: true,
        show_assignments: true,
        show_boundaries: false,
        show_elbow: false,
        animation_duration: 1500,
        interactive: true
      },
      performanceTips: [
        'k-means++ initialization generally gives better results than random',
        'Elbow method visualization helps determine optimal K value',
        'Higher cluster std values create more overlapping clusters'
      ]
    },
    
    implementations: {
      python: {
        version: "3.9",
        libraries: ['scikit-learn@1.0.2', 'numpy@1.21.5', 'matplotlib@3.5.1'],
        code: `# Python implementation using scikit-learn
  from sklearn.cluster import KMeans
  from sklearn.datasets import make_blobs
  import matplotlib.pyplot as plt

  # Generate sample data
  X, y_true = make_blobs(n_samples=300, centers=3, 
                        cluster_std=0.8, random_state=42)

  # Create and fit model
  kmeans = KMeans(n_clusters=3, init='k-means++', 
                  max_iter=300, random_state=42)
  kmeans.fit(X)
  y_pred = kmeans.predict(X)

  # Get cluster centers
  centers = kmeans.cluster_centers_

  # Plot results
  plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
  plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
  plt.title('K-Means Clustering')
  plt.show()`,
        timeComplexity: "O(n*K*I*d) where n is samples, K is clusters, I is iterations, d is dimensions",
        spaceComplexity: "O(n*d + K*d)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For large datasets, consider using MiniBatchKMeans. The k-means++ initialization generally gives better results than random initialization."
      },
      r: {
        version: "4.2",
        libraries: ['stats@4.2.0', 'factoextra@1.0.7'],
        code: `# R implementation
  # Generate sample data
  set.seed(42)
  data <- matrix(rnorm(300*2), ncol=2)
  data[1:100,] <- data[1:100,] + 3
  data[101:200,] <- data[101:200,] - 3

  # Perform k-means clustering
  kmeans_result <- kmeans(data, centers=3, nstart=20)

  # Plot results
  library(factoextra)
  fviz_cluster(kmeans_result, data = data, 
              palette = "jco", 
              geom = "point",
              ellipse.type = "convex",
              ggtheme = theme_bw())`,
        timeComplexity: "O(n*K*I*d)",
        spaceComplexity: "O(n*d + K*d)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "Use nstart > 1 to run the algorithm multiple times with different initial centroids and select the best result. The factoextra package provides excellent visualization tools."
      },
      javascript: {
        version: "1.7.0",
        libraries: ['ml.js@0.12.0'],
        code: `// JavaScript implementation using ml.js
  import { KMeans } from 'ml-kmeans';

  // Sample data
  const data = [
    [1, 2], [1, 4], [1, 0],
    [10, 2], [10, 4], [10, 0],
    [20, 2], [20, 4], [20, 0]
  ];

  // Perform k-means clustering
  const result = KMeans(data, 3, { initialization: 'kmeans++' });

  // Output results
  console.log('Centroids:', result.centroids);
  console.log('Cluster assignments:', result.clusters);

  // Calculate inertia (within-cluster sum of squares)
  let inertia = 0;
  for (let i = 0; i < data.length; i++) {
    const centroid = result.centroids[result.clusters[i]];
    inertia += Math.pow(data[i][0] - centroid[0], 2) + 
              Math.pow(data[i][1] - centroid[1], 2);
  }
  console.log('Inertia:', inertia);`,
        timeComplexity: "O(n*K*I*d)",
        spaceComplexity: "O(n*d + K*d)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "The ml.js implementation includes multiple initialization methods. For large datasets, consider sampling or using approximate methods."
      },
      cpp: {
        version: "17",
        libraries: ['mlpack@3.4.2'],
        code: `// C++ implementation using mlpack
  #include <mlpack.hpp>
  #include <iostream>

  using namespace mlpack;
  using namespace arma;

  int main() {
    // Generate sample data
    mat data(2, 300);
    data.randn();
    data.cols(0, 99) += mat(2, 100, fill::ones) * 3;
    data.cols(100, 199) -= mat(2, 100, fill::ones) * 3;
    
    // Perform k-means clustering
    mat centroids;
    Row<size_t> assignments;
    kmeans::KMeans<> k;
    k.Cluster(data, 3, assignments, centroids);
    
    // Output results
    std::cout << "Centroids:" << std::endl;
    std::cout << centroids << std::endl;
    
    return 0;
  }`,
        timeComplexity: "O(n*K*I*d)",
        spaceComplexity: "O(n*d + K*d)",
        author: {
          name: "Numerical Computing Team"
        },
        lastUpdated: "2025-08-29",
        notes: "mlpack provides multiple k-means variants including allow_empty_clusters and kill_empty_clusters options. For very large datasets, consider using the dual-tree algorithm."
      }
    },

    prosCons: {
      strengths: [
        'Simple and easy to implement',
        'Efficient and scalable to large datasets',
        'Guaranteed convergence',
        'Adapts easily to new data'
      ],
      weaknesses: [
        'Sensitive to initial centroid positions',
        'Requires specifying number of clusters K',
        'Sensitive to outliers',
        'Assumes spherical clusters of similar size'
      ]
    },
    
    hyperparameters: [
      {
        name: 'n_clusters',
        type: 'range',
        min: 2,
        max: 10,
        step: 1,
        default: 3,
        description: 'Number of clusters to form'
      },
      {
        name: 'init',
        type: 'select',
        options: ['k-means++', 'random'],
        default: 'k-means++',
        description: 'Initialization method for centroids'
      },
      {
        name: 'max_iter',
        type: 'range',
        min: 100,
        max: 1000,
        step: 50,
        default: 300,
        description: 'Maximum number of iterations'
      }
    ],
    
    useCases: [
      {
        title: 'Customer Segmentation',
        description: 'Grouping customers based on purchasing behavior for targeted marketing.',
        dataset: 'Mall Customer Segmentation'
      },
      {
        title: 'Image Compression',
        description: 'Reducing color space by grouping similar colors together.',
        dataset: 'Standard Image Datasets'
      }
    ],
    
    comparisons: [
      {
        algorithm: 'Hierarchical Clustering',
        comparison: 'K-means requires specifying K in advance while hierarchical clustering builds a dendrogram showing all possible clusters'
      },
      {
        algorithm: 'DBSCAN',
        comparison: 'K-means assumes spherical clusters while DBSCAN can find arbitrarily shaped clusters and doesn\'t require specifying K'
      }
    ],
    
    quiz: [
      {
        question: 'What is the main objective of k-means clustering?',
        options: [
          'To minimize the distance between points in different clusters',
          'To maximize the distance between points in the same cluster',
          'To minimize the within-cluster sum of squares',
          'To maximize the between-cluster variance'
        ],
        correct: 2,
        explanation: 'K-means aims to minimize the within-cluster sum of squares (variance), which is equivalent to minimizing the distance between points and their cluster centroids.'
      },
      {
        question: 'Which initialization method generally gives better results for k-means?',
        options: [
          'Random initialization',
          'k-means++',
          'Farthest-first traversal',
          'All methods give similar results'
        ],
        correct: 1,
        explanation: 'k-means++ initialization spreads out the initial centroids, leading to better and more consistent results compared to random initialization.'
      }
    ],
    
    projects: [
      {
        title: 'Customer Segmentation Analysis',
        description: 'Use k-means clustering to segment customers based on their purchasing behavior.',
        steps: [
          'Load and preprocess customer data',
          'Standardize features for clustering',
          'Use elbow method to determine optimal K',
          'Apply k-means clustering',
          'Analyze and interpret customer segments',
          'Visualize clusters using PCA'
        ],
        difficulty: 'intermediate',
        xp: 350
      }
    ]
  },
  
  // Hierarchical Clustering AlgoData
  {
    id: 'hierarchical-clustering',
    title: 'Hierarchical Clustering',
    category: 'unsupervised',
    difficulty: 'beginner',
    tags: ['Clustering', 'Unsupervised', 'Dendrogram'],
    description: 'Builds a hierarchy of clusters either through a bottom-up (agglomerative) or top-down (divisive) approach, represented as a tree structure called a dendrogram.',
    icon: 'sitemap',
    lastUpdated: '2025-08-29',
    popularity: 0.75,
    
    concept: {
      overview: 'Hierarchical clustering creates a tree of clusters called a dendrogram that shows the hierarchical relationship between all data points, allowing analysis at different levels of granularity.',
      analogy: 'Like building a family tree where individuals are grouped into families, families into clans, and clans into tribes based on genetic similarity.',
      history: 'Developed in the 1950s for numerical taxonomy in biology. Key algorithms include single linkage (1951), complete linkage (1953), and Ward\'s method (1963).',
      mathematicalFormulation: {
        linkageMethods: [
          {
            name: 'Single Linkage',
            formula: 'd(A,B) = min{d(a,b): a∈A, b∈B}',
            description: 'Distance between clusters is the minimum distance between any two points'
          },
          {
            name: 'Complete Linkage',
            formula: 'd(A,B) = max{d(a,b): a∈A, b∈B}',
            description: 'Distance between clusters is the maximum distance between any two points'
          },
          {
            name: 'Average Linkage',
            formula: 'd(A,B) = (1/|A||B|) ΣΣ d(a,b)',
            description: 'Distance between clusters is the average distance between all pairs'
          },
          {
            name: 'Ward\'s Method',
            formula: 'd(A,B) = Δ(AB) = ESS(AB) - [ESS(A) + ESS(B)]',
            description: 'Minimizes the increase in within-cluster variance after merging'
          }
        ],
        algorithmSteps: [
          'Start with each point as its own cluster',
          'Find the two closest clusters and merge them',
          'Update the distance matrix',
          'Repeat until all points are in one cluster'
        ]
      },
      assumptions: [
        'The chosen distance metric appropriately captures similarity',
        'The linkage method is suitable for the data structure',
        'The dendrogram provides meaningful interpretation of data hierarchy'
      ]
    },
    
    visualization: {
      visualizerKey: 'hierarchical-clustering',
      defaultType: 'dendrogram',
      description: 'Interactive visualization of hierarchical clustering. Explore how clusters are merged at different distances and cut the dendrogram to obtain flat clusters.',
      instructions: [
        'Adjust the linkage method to see different clustering behaviors',
        'Drag the cutoff line on the dendrogram to change the number of clusters',
        'Observe how clusters are merged at different heights',
        'Compare different distance metrics'
      ],
      showTypeSelector: true,
      types: [
        { 
          value: 'dendrogram', 
          label: 'Dendrogram', 
          description: 'Tree structure showing cluster merging',
          default: true 
        },
        { 
          value: 'cluster-formation', 
          label: 'Cluster Formation',
          description: 'Step-by-step visualization of merging process'
        },
        { 
          value: 'heatmap-dendrogram', 
          label: 'Heatmap with Dendrogram',
          description: 'Combines heatmap with clustering tree'
        },
        { 
          value: 'all', 
          label: 'Complete View',
          description: 'All visualization elements together'
        },
        { 
          value: 'interactive-cut', 
          label: 'Interactive Cutting',
          description: 'Allows dynamic cutting of the dendrogram'
        }
      ],
      parameters: {
        n_samples: 100,
        n_clusters: 3,
        linkage: 'ward',
        distance_metric: 'euclidean',
        show_dendrogram: true,
        show_clusters: true,
        show_heatmap: false,
        show_cut_line: true,
        animation_duration: 2000,
        interactive: true
      },
      performanceTips: [
        'Ward\'s method works well with Euclidean distance',
        'Single linkage can create "chaining" effect (long elongated clusters)',
        'For large datasets, consider using efficient implementations or sampling'
      ]
    },
    
    implementations: {
      python: {
        version: "3.9",
        libraries: ['scikit-learn@1.0.2', 'scipy@1.7.3', 'matplotlib@3.5.1'],
        code: `# Python implementation using scikit-learn and SciPy
  from sklearn.datasets import make_blobs
  from sklearn.cluster import AgglomerativeClustering
  from scipy.cluster.hierarchy import dendrogram, linkage
  import matplotlib.pyplot as plt

  # Generate sample data
  X, y_true = make_blobs(n_samples=100, centers=3, 
                        cluster_std=0.8, random_state=42)

  # Perform hierarchical clustering
  clustering = AgglomerativeClustering(n_clusters=3, 
                                      linkage='ward')
  y_pred = clustering.fit_predict(X)

  # Create linkage matrix for dendrogram
  Z = linkage(X, 'ward')

  # Plot dendrogram
  plt.figure(figsize=(10, 5))
  plt.title('Hierarchical Clustering Dendrogram')
  plt.xlabel('Sample index')
  plt.ylabel('Distance')
  dendrogram(Z)
  plt.show()`,
        timeComplexity: "O(n³) for most implementations, O(n²) for efficient implementations",
        spaceComplexity: "O(n²)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For large datasets, consider using efficient implementations or sampling methods. The SciPy implementation provides multiple linkage methods and distance metrics."
      },
      r: {
        version: "4.2",
        libraries: ['stats@4.2.0', 'factoextra@1.0.7'],
        code: `# R implementation
  # Generate sample data
  set.seed(42)
  data <- matrix(rnorm(100*2), ncol=2)
  data[1:33,] <- data[1:33,] + 3
  data[34:66,] <- data[34:66,] - 2

  # Compute distance matrix
  dist_matrix <- dist(data)

  # Perform hierarchical clustering
  hc <- hclust(dist_matrix, method = "ward.D2")

  # Plot dendrogram
  plot(hc, cex = 0.6, hang = -1)
  rect.hclust(hc, k = 3, border = 2:4)

  # Cut dendrogram to get clusters
  clusters <- cutree(hc, k = 3)

  # Enhanced visualization with factoextra
  library(factoextra)
  fviz_dend(hc, k = 3, cex = 0.5, k_colors = "jco")`,
        timeComplexity: "O(n³)",
        spaceComplexity: "O(n²)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "The factoextra package provides excellent visualization tools for hierarchical clustering. Note that Ward's method in R is called 'ward.D' or 'ward.D2'."
      },
      javascript: {
        version: "1.7.0",
        libraries: ['ml.js@0.12.0'],
        code: `// JavaScript implementation using ml.js
  import { agnes } from 'ml-hclust';

  // Sample data
  const data = [
    [1, 2], [1, 4], [1, 0],
    [10, 2], [10, 4], [10, 0],
    [20, 2], [20, 4], [20, 0]
  ];

  // Perform hierarchical clustering
  const tree = agnes(data, {
    method: 'ward',
    isDistanceMatrix: false
  });

  // Get clusters by cutting the tree
  const clusters = tree.cut(3);

  // Output results
  console.log('Cluster assignments:', clusters);

  // Get the full dendrogram for visualization
  console.log('Dendrogram:', tree.getDendrogram());`,
        timeComplexity: "O(n³)",
        spaceComplexity: "O(n²)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "The ml.js implementation includes multiple linkage methods. For large datasets, consider using more efficient algorithms or sampling techniques."
      },
      cpp: {
        version: "17",
        libraries: ['mlpack@3.4.2'],
        code: `// C++ implementation using mlpack
  #include <mlpack.hpp>
  #include <iostream>

  using namespace mlpack;
  using namespace arma;

  int main() {
    // Generate sample data
    mat data(2, 100);
    data.randn();
    data.cols(0, 32) += mat(2, 33, fill::ones) * 3;
    data.cols(33, 65) -= mat(2, 33, fill::ones) * 2;
    
    // Perform hierarchical clustering
    Row<size_t> assignments;
    cluster::HDBSCAN<> hdbscan;
    hdbscan.Cluster(data, assignments);
    
    // For standard hierarchical clustering, we might need to implement
    // our own or use a different library as mlpack focuses on HDBSCAN
    
    std::cout << "Cluster assignments:" << std::endl;
    std::cout << assignments << std::endl;
    
    return 0;
  }`,
        timeComplexity: "O(n³)",
        spaceComplexity: "O(n²)",
        author: {
          name: "Numerical Computing Team"
        },
        lastUpdated: "2025-08-29",
        notes: "mlpack focuses on HDBSCAN rather than standard hierarchical clustering. For standard implementations, consider other libraries or custom implementation."
      }
    },

    prosCons: {
      strengths: [
        'Does not require specifying number of clusters in advance',
        'Provides hierarchical relationship between clusters',
        'Visual dendrogram helps interpret results',
        'Flexible with different distance metrics and linkage methods'
      ],
      weaknesses: [
        'Computationally expensive for large datasets',
        'Sensitive to noise and outliers',
        'Different linkage methods can give very different results',
        'Once a merge is done, it cannot be undone'
      ]
    },
    
    hyperparameters: [
      {
        name: 'linkage',
        type: 'select',
        options: ['ward', 'complete', 'average', 'single'],
        default: 'ward',
        description: 'Linkage criterion to use'
      },
      {
        name: 'distance_metric',
        type: 'select',
        options: ['euclidean', 'manhattan', 'cosine', 'correlation'],
        default: 'euclidean',
        description: 'Distance metric to use'
      },
      {
        name: 'n_clusters',
        type: 'range',
        min: 2,
        max: 10,
        step: 1,
        default: 3,
        description: 'Number of clusters to extract from dendrogram'
      }
    ],
    
    useCases: [
      {
        title: 'Gene Expression Analysis',
        description: 'Grouping genes with similar expression patterns to identify functional relationships.',
        dataset: 'Gene Expression Microarray Data'
      },
      {
        title: 'Document Clustering',
        description: 'Organizing documents into a hierarchy of topics and subtopics.',
        dataset: 'News Articles Corpus'
      }
    ],
    
    comparisons: [
      {
        algorithm: 'K-Means',
        comparison: 'Hierarchical clustering doesn\'t require specifying K in advance and provides a dendrogram, but is computationally more expensive'
      },
      {
        algorithm: 'DBSCAN',
        comparison: 'Hierarchical clustering can find nested clusters but is more sensitive to noise and parameter choices'
      }
    ],
    
    quiz: [
      {
        question: 'Which linkage method in hierarchical clustering minimizes the increase in total within-cluster variance?',
        options: [
          'Single linkage',
          'Complete linkage',
          'Average linkage',
          'Ward\'s method'
        ],
        correct: 3,
        explanation: 'Ward\'s method minimizes the increase in total within-cluster variance after merging, making it similar to k-means objectives.'
      },
      {
        question: 'What is the main advantage of hierarchical clustering over partitioning methods like k-means?',
        options: [
          'It\'s faster computationally',
          'It doesn\'t require specifying the number of clusters in advance',
          'It always gives better results',
          'It works better with high-dimensional data'
        ],
        correct: 1,
        explanation: 'The main advantage is that hierarchical clustering doesn\'t require specifying the number of clusters in advance, instead providing a dendrogram that shows relationships at all levels.'
      }
    ],
    
    projects: [
      {
        title: 'Gene Expression Clustering',
        description: 'Use hierarchical clustering to group genes with similar expression patterns.',
        steps: [
          'Load and preprocess gene expression data',
          'Compute distance matrix using appropriate metric',
          'Apply hierarchical clustering with different linkage methods',
          'Visualize results as a dendrogram',
          'Cut dendrogram to obtain clusters at different levels',
          'Interpret biological significance of clusters'
        ],
        difficulty: 'advanced',
        xp: 450
      }
    ]
  },

  // Gaussian Mixture Models AlgoData
  {
    id: 'gaussian-mixture-models',
    title: 'Gaussian Mixture Models (GMM)',
    category: 'unsupervised',
    difficulty: 'intermediate',
    tags: ['Clustering', 'Probability', 'Unsupervised', 'EM Algorithm'],
    description: 'A probabilistic model that assumes all data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters.',
    icon: 'chart-pie',
    lastUpdated: '2025-08-29',
    popularity: 0.68,
    
    concept: {
      overview: 'Gaussian Mixture Models represent a composite distribution whereby points are drawn from one of several Gaussian distributions, each with its own mean and covariance, with the mixture proportions adding to one.',
      analogy: 'Imagine trying to separate a mixture of three different types of beans based on their size and weight. Each bean type follows a different normal distribution for these measurements.',
      history: 'The Expectation-Maximization algorithm for GMMs was formalized by Dempster, Laird, and Rubin in 1977, though the concept dates back to earlier work in the 19th century.',
      mathematicalFormulation: {
        probabilityDensity: 'p(x) = ΣπₖN(x|μₖ,Σₖ)',
        variables: [
          { symbol: 'πₖ', description: 'Mixing coefficient for component k (0≤πₖ≤1, Σπₖ=1)' },
          { symbol: 'μₖ', description: 'Mean of component k' },
          { symbol: 'Σₖ', description: 'Covariance matrix of component k' },
          { symbol: 'N(x|μₖ,Σₖ)', description: 'Multivariate Gaussian distribution' }
        ],
        algorithmSteps: [
          'Initialize parameters (means, covariances, and mixing coefficients)',
          'E-step: Compute responsibilities (probability each point belongs to each component)',
          'M-step: Re-estimate parameters using current responsibilities',
          'Repeat until convergence of log-likelihood'
        ]
      },
      assumptions: [
        'Data points are generated from a mixture of Gaussian distributions',
        'Each component has its own mean and covariance',
        'The number of components K is known or can be estimated'
      ]
    },
    
    visualization: {
      visualizerKey: 'gaussian-mixture-models',
      defaultType: 'probability-surface',
      description: 'Interactive visualization of Gaussian Mixture Models. Observe how the EM algorithm iteratively fits Gaussian components to the data.',
      instructions: [
        'Adjust the number of components to see different mixture models',
        'Watch the EM algorithm converge to the optimal parameters',
        'Toggle between different covariance types',
        'Observe the soft assignments (probabilities) of points to components'
      ],
      showTypeSelector: true,
      types: [
        { 
          value: 'probability-surface', 
          label: 'Probability Surface', 
          description: 'Shows the probability density function of the mixture model',
          default: true 
        },
        { 
          value: 'component-visualization', 
          label: 'Component Visualization',
          description: 'Shows individual Gaussian components'
        },
        { 
          value: 'em-steps', 
          label: 'EM Algorithm Steps',
          description: 'Step-by-step visualization of Expectation-Maximization'
        },
        { 
          value: 'all', 
          label: 'Complete View',
          description: 'All visualization elements together'
        },
        { 
          value: 'with-responsibilities', 
          label: 'With Responsibilities',
          description: 'Shows soft assignments of points to components'
        }
      ],
      parameters: {
        n_samples: 300,
        n_components: 3,
        covariance_type: 'full',
        max_iter: 100,
        tol: 0.001,
        random_state: 42,
        show_components: true,
        show_probability: true,
        show_em_steps: false,
        show_responsibilities: false,
        animation_duration: 2000,
        interactive: true
      },
      performanceTips: [
        'Full covariance allows more flexibility but requires more parameters',
        'Diagonal covariance is more constrained but faster to compute',
        'Spherical covariance assumes all components are circular'
      ]
    },
    
    implementations: {
      python: {
        version: "3.9",
        libraries: ['scikit-learn@1.0.2', 'numpy@1.21.5', 'matplotlib@3.5.1'],
        code: `# Python implementation using scikit-learn
  from sklearn.mixture import GaussianMixture
  from sklearn.datasets import make_blobs
  import matplotlib.pyplot as plt
  import numpy as np

  # Generate sample data
  X, y_true = make_blobs(n_samples=300, centers=3, 
                        cluster_std=0.8, random_state=42)

  # Create and fit model
  gmm = GaussianMixture(n_components=3, covariance_type='full',
                        max_iter=100, random_state=42)
  gmm.fit(X)
  y_pred = gmm.predict(X)

  # Get parameters
  means = gmm.means_
  covariances = gmm.covariances_

  # Plot results
  plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
  plt.scatter(means[:, 0], means[:, 1], c='red', s=200, alpha=0.75)
  plt.title('Gaussian Mixture Model Clustering')
  plt.show()

  # Get probabilities (responsibilities)
  probs = gmm.predict_proba(X)
  print('Sample probabilities:', probs[0])`,
        timeComplexity: "O(n*K*I*d³) for full covariance, O(n*K*I*d) for diagonal covariance",
        spaceComplexity: "O(n*K + K*d²)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For high-dimensional data, consider using diagonal covariance to reduce computational complexity. The Bayesian Information Criterion (BIC) can help select the optimal number of components."
      },
      r: {
        version: "4.2",
        libraries: ['mclust@6.0.0'],
        code: `# R implementation using mclust
  # Generate sample data
  set.seed(42)
  data <- matrix(rnorm(300*2), ncol=2)
  data[1:100,] <- data[1:100,] + 3
  data[101:200,] <- data[101:200,] - 2

  # Fit Gaussian Mixture Model
  library(mclust)
  gmm <- Mclust(data, G = 3, modelNames = "VVV")

  # Print summary
  summary(gmm)

  # Plot results
  plot(gmm, what = "classification")
  plot(gmm, what = "uncertainty")

  # Get probabilities
  probs <- gmm$z
  print('Sample probabilities:', head(probs))`,
        timeComplexity: "O(n*K*I*d³)",
        spaceComplexity: "O(n*K + K*d²)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "The mclust package provides sophisticated model selection. VVV indicates varying volume, shape, and orientation (full covariance). Use BIC to select the optimal number of components."
      },
      javascript: {
        version: "1.7.0",
        libraries: ['ml.js@0.12.0'],
        code: `// JavaScript implementation using ml.js
  import { GMM } from 'ml-gmm';

  // Sample data
  const data = [
    [1, 2], [1, 4], [1, 0],
    [10, 2], [10, 4], [10, 0],
    [20, 2], [20, 4], [20, 0]
  ];

  // Create and fit Gaussian Mixture Model
  const options = {
    gaussians: 3,
    tolerance: 1e-3,
    maxIterations: 100
  };

  const gmm = new GMM(options);
  gmm.train(data);

  // Predict clusters
  const predictions = gmm.predict(data);
  console.log('Cluster assignments:', predictions);

  // Get probabilities
  const probabilities = gmm.predictProbability(data);
  console.log('Probabilities:', probabilities[0]);`,
        timeComplexity: "O(n*K*I*d³)",
        spaceComplexity: "O(n*K + K*d²)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "The ml.js implementation includes the EM algorithm for parameter estimation. For high-dimensional data, consider dimensionality reduction techniques first."
      },
      cpp: {
        version: "17",
        libraries: ['mlpack@3.4.2'],
        code: `// C++ implementation using mlpack
  #include <mlpack.hpp>
  #include <iostream>

  using namespace mlpack;
  using namespace arma;

  int main() {
    // Generate sample data
    mat data(2, 300);
    data.randn();
    data.cols(0, 99) += mat(2, 100, fill::ones) * 3;
    data.cols(100, 199) -= mat(2, 100, fill::ones) * 2;
    
    // Create and fit Gaussian Mixture Model
    gmm::GMM gmm(3, 2); // 3 Gaussians, 2 dimensions
    double likelihood = gmm.Train(data, 10); // 10 iterations
    
    // Get parameters
    std::vector<arma::vec> means = gmm.Means();
    std::vector<arma::mat> covariances = gmm.Covariances();
    
    // Predict clusters
    arma::Row<size_t> assignments;
    gmm.Classify(data, assignments);
    
    std::cout << "Log-likelihood: " << likelihood << std::endl;
    std::cout << "Means: " << std::endl;
    for (const auto& mean : means) {
      std::cout << mean.t();
    }
    
    return 0;
  }`,
        timeComplexity: "O(n*K*I*d³)",
        spaceComplexity: "O(n*K + K*d²)",
        author: {
          name: "Numerical Computing Team"
        },
        lastUpdated: "2025-08-29",
        notes: "mlpack provides efficient implementations of GMMs. For high-dimensional data, consider using diagonal covariance matrices to reduce computational complexity."
      }
    },

    prosCons: {
      strengths: [
        'Provides soft clustering (probabilistic assignments)',
        'Flexible cluster shapes (depending on covariance type)',
        'Works well with overlapping clusters',
        'Based on solid statistical foundation'
      ],
      weaknesses: [
        'Sensitive to initialization',
        'Can converge to local maxima',
        'Computationally expensive for large datasets',
        'Requires specifying number of components'
      ]
    },
    
    hyperparameters: [
      {
        name: 'n_components',
        type: 'range',
        min: 1,
        max: 10,
        step: 1,
        default: 3,
        description: 'Number of mixture components'
      },
      {
        name: 'covariance_type',
        type: 'select',
        options: ['full', 'tied', 'diag', 'spherical'],
        default: 'full',
        description: 'Type of covariance parameters to use'
      },
      {
        name: 'max_iter',
        type: 'range',
        min: 10,
        max: 500,
        step: 10,
        default: 100,
        description: 'Maximum number of EM iterations'
      }
    ],
    
    useCases: [
      {
        title: 'Anomaly Detection',
        description: 'Identifying unusual patterns that don\'t conform to expected behavior by modeling normal behavior as a mixture of Gaussians.',
        dataset: 'Network Intrusion Detection'
      },
      {
        title: 'Image Segmentation',
        description: 'Separating foreground from background or identifying different regions in an image based on color distributions.',
        dataset: 'Natural Images'
      }
    ],
    
    comparisons: [
      {
        algorithm: 'K-Means',
        comparison: 'GMM provides soft assignments and can model elliptical clusters while k-means provides hard assignments and assumes spherical clusters'
      },
      {
        algorithm: 'DBSCAN',
        comparison: 'GMM assumes parametric distributions while DBSCAN is non-parametric and can find arbitrarily shaped clusters'
      }
    ],
    
    quiz: [
      {
        question: 'What does the EM algorithm stand for in the context of Gaussian Mixture Models?',
        options: [
          'Error Minimization',
          'Expectation-Maximization',
          'Efficient Modeling',
          'Estimation-Measurement'
        ],
        correct: 1,
        explanation: 'EM stands for Expectation-Maximization, which is the algorithm used to estimate the parameters of GMMs by iteratively improving the likelihood.'
      },
      {
        question: 'What is the main advantage of GMM over k-means clustering?',
        options: [
          'It\'s computationally faster',
          'It provides soft (probabilistic) assignments',
          'It always converges to the global optimum',
          'It doesn\'t require specifying the number of clusters'
        ],
        correct: 1,
        explanation: 'The main advantage is that GMM provides soft assignments, meaning each point has a probability of belonging to each cluster, rather than a hard assignment.'
      }
    ],
    
    projects: [
      {
        title: 'Anomaly Detection System',
        description: 'Build an anomaly detection system using Gaussian Mixture Models to identify unusual patterns in data.',
        steps: [
          'Load and preprocess the dataset',
          'Fit a GMM to model normal behavior',
          'Calculate likelihood scores for each data point',
          'Set threshold to identify anomalies',
          'Evaluate detection performance',
          'Visualize normal vs anomalous patterns'
        ],
        difficulty: 'advanced',
        xp: 500
      }
    ]
  },

  // Principal Component Analysis AlgoData
  {
    id: 'pca',
    title: 'Principal Component Analysis',
    category: 'unsupervised',
    difficulty: 'intermediate',
    tags: ['Dimensionality Reduction', 'Unsupervised', 'Feature Extraction'],
    description: 'A statistical technique that transforms correlated variables into a set of uncorrelated variables called principal components, ordered by the amount of variance they explain.',
    icon: 'compress-alt',
    lastUpdated: '2025-08-29',
    popularity: 0.78,
    
    concept: {
      overview: 'PCA identifies patterns in data and expresses the data in such a way as to highlight their similarities and differences. It reduces dimensionality while preserving as much variability as possible.',
      analogy: 'Imagine taking a photo of a 3D object from different angles. PCA finds the "best" angles that capture the most important features of the object.',
      history: 'Invented by Karl Pearson in 1901 and later developed by Harold Hotelling in the 1930s. Became widely used with the advent of computers.',
      mathematicalFormulation: {
        process: [
          'Standardize the data',
          'Compute covariance matrix',
          'Calculate eigenvectors and eigenvalues',
          'Sort eigenvectors by eigenvalues in descending order',
          'Project data onto the new feature space'
        ],
        keyConcepts: [
          {
            name: 'Eigenvectors',
            description: 'Directions of maximum variance in the data'
          },
          {
            name: 'Eigenvalues',
            description: 'Magnitude of variance along each eigenvector'
          },
          {
            name: 'Explained Variance Ratio',
            description: 'Percentage of variance explained by each component'
          }
        ]
      },
      assumptions: [
        'Variables are measured on continuous scales',
        'Variables are correlated (to some degree)',
        'Large sample size relative to number of variables',
        'Data should be standardized'
      ]
    },
    
    visualization: {
      visualizerKey: 'pca',
      defaultType: 'variance-explained',
      description: 'Interactive visualization of PCA algorithm. Explore how data is transformed and how variance is distributed across components.',
      instructions: [
        'Adjust number of components to see reconstruction quality',
        'Rotate 3D plot to view data from different angles',
        'Toggle between original and transformed data views',
        'Explore scree plot to determine optimal component count'
      ],
      showTypeSelector: true,
      types: [
        { 
          value: 'variance-explained', 
          label: 'Variance Explained', 
          description: 'Shows how much variance each component captures',
          default: true 
        },
        { 
          value: 'biplot', 
          label: 'Biplot',
          description: 'Combines data points and variable contributions'
        },
        { 
          value: 'scree-plot', 
          label: 'Scree Plot',
          description: 'Visualizes eigenvalues to help determine component count'
        },
        { 
          value: 'reconstruction', 
          label: 'Reconstruction',
          description: 'Shows original vs. reconstructed data'
        },
        { 
          value: 'all', 
          label: 'Complete View',
          description: 'All visualization elements together'
        }
      ],
      parameters: {
        n_samples: 200,
        n_features: 10,
        n_components: 2,
        variance_explained: 0.95,
        show_original: false,
        show_transformed: true,
        show_vectors: true,
        show_variance: true,
        show_reconstruction: false,
        animation_duration: 1500,
        interactive: true
      },
      performanceTips: [
        '3D visualization works best with 50-200 data points',
        'Biplots help understand variable contributions to components',
        'Scree plots show the "elbow" point for optimal component selection'
      ]
    },
    
    implementations: {
      python: {
        version: "3.9",
        libraries: ['scikit-learn@1.0.2', 'numpy@1.21.5', 'pandas@1.3.5'],
        code: `# Python implementation using scikit-learn
  from sklearn.decomposition import PCA
  from sklearn.datasets import load_iris
  from sklearn.preprocessing import StandardScaler
  import matplotlib.pyplot as plt

  # Load sample data
  data = load_iris()
  X, y = data.data, data.target

  # Standardize the data
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  # Apply PCA
  pca = PCA(n_components=2)
  X_pca = pca.fit_transform(X_scaled)

  # Print results
  print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
  print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.3f}")

  # Plot results
  plt.figure(figsize=(8, 6))
  for i, target_name in enumerate(data.target_names):
      plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], alpha=0.8, label=target_name)
  plt.legend()
  plt.title('PCA of IRIS dataset')
  plt.xlabel('Principal Component 1')
  plt.ylabel('Principal Component 2')
  plt.show()`,
        timeComplexity: "O(n³) for full SVD, O(n²k) for truncated SVD",
        spaceComplexity: "O(n²)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For very large datasets, consider using IncrementalPCA or RandomizedPCA for better memory efficiency."
      },
      r: {
        version: "4.2",
        libraries: ['stats@4.2.0', 'factoextra@1.0.7'],
        code: `# R implementation
  library(factoextra)

  # Load and standardize data
  data(iris)
  iris_scaled <- scale(iris[, 1:4])

  # Apply PCA
  pca_result <- prcomp(iris_scaled, center = TRUE, scale. = TRUE)

  # Print summary
  summary(pca_result)

  # Visualize
  fviz_eig(pca_result) # Scree plot
  fviz_pca_ind(pca_result, col.ind = iris$Species) # Individuals plot
  fviz_pca_var(pca_result) # Variables plot

  # Biplot
  fviz_pca_biplot(pca_result, col.ind = iris$Species)`,
        timeComplexity: "O(n³)",
        spaceComplexity: "O(n²)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "The factoextra package provides excellent visualization functions for PCA results. For large datasets, consider the irlba package for partial SVD."
      },
      javascript: {
        version: "1.7.0",
        libraries: ['ml.js@0.12.0'],
        code: `// JavaScript implementation using ml.js
  import { PCA } from 'ml.js';

  // Sample data (4D iris-like data)
  const data = [
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [4.7, 3.2, 1.3, 0.2],
    [4.6, 3.1, 1.5, 0.2],
    [5.0, 3.6, 1.4, 0.2],
    [7.0, 3.2, 4.7, 1.4],
    [6.4, 3.2, 4.5, 1.5],
    [6.9, 3.1, 4.9, 1.5],
    [5.5, 2.3, 4.0, 1.3],
    [6.5, 2.8, 4.6, 1.5]
  ];

  // Apply PCA
  const pca = new PCA();
  const result = pca.fitTransform(data, { nComponents: 2 });

  console.log('Transformed data:', result.data);
  console.log('Explained variance:', result.explainedVariance);`,
        timeComplexity: "O(n³)",
        spaceComplexity: "O(n²)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For browser-based applications, consider using the precomputed SVD to avoid the computational cost of PCA on large datasets."
      },
      cpp: {
        version: "17",
        libraries: ['Eigen@3.4.0'],
        code: `// C++ implementation using Eigen
  #include <Eigen/Dense>
  #include <iostream>

  using namespace Eigen;

  int main() {
    // Sample data (rows: samples, columns: features)
    MatrixXd data(10, 4);
    data << 5.1, 3.5, 1.4, 0.2,
            4.9, 3.0, 1.4, 0.2,
            4.7, 3.2, 1.3, 0.2,
            4.6, 3.1, 1.5, 0.2,
            5.0, 3.6, 1.4, 0.2,
            7.0, 3.2, 4.7, 1.4,
            6.4, 3.2, 4.5, 1.5,
            6.9, 3.1, 4.9, 1.5,
            5.5, 2.3, 4.0, 1.3,
            6.5, 2.8, 4.6, 1.5;

    // Center the data
    VectorXd mean = data.colwise().mean();
    MatrixXd centered = data.rowwise() - mean.transpose();

    // Compute covariance matrix
    MatrixXd cov = (centered.adjoint() * centered) / (data.rows() - 1);

    // Compute eigenvectors and eigenvalues
    SelfAdjointEigenSolver<MatrixXd> es(cov);
    MatrixXd eigenvectors = es.eigenvectors();
    VectorXd eigenvalues = es.eigenvalues();

    // Sort eigenvectors by eigenvalues (descending)
    std::vector<std::pair<double, VectorXd>> eigen_pairs;
    for (int i = 0; i < eigenvalues.size(); ++i) {
      eigen_pairs.push_back(std::make_pair(eigenvalues(i), eigenvectors.col(i)));
    }
    std::sort(eigen_pairs.begin(), eigen_pairs.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Project data onto principal components
    MatrixXd projection_matrix(eigen_pairs[0].second.size(), 2);
    for (int i = 0; i < 2; ++i) {
      projection_matrix.col(i) = eigen_pairs[i].second;
    }
    MatrixXd transformed = centered * projection_matrix;

    std::cout << "Transformed data:\\n" << transformed << std::endl;
    return 0;
  }`,
        timeComplexity: "O(n³)",
        spaceComplexity: "O(n²)",
        author: {
          name: "Numerical Computing Team"
        },
        lastUpdated: "2025-08-29",
        notes: "For very large matrices, consider using iterative methods like the power method or randomized SVD for efficiency."
      }
    },

    prosCons: {
      strengths: [
        'Reduces dimensionality while preserving variance',
        'Removes correlation between features',
        'Improves algorithm performance by reducing noise',
        'Helps visualize high-dimensional data'
      ],
      weaknesses: [
        'Results can be hard to interpret',
        'Sensitive to feature scaling',
        'Assumes linear relationships in data',
        'Information loss when reducing dimensions'
      ]
    },
    
    hyperparameters: [
      {
        name: 'n_components',
        type: 'range',
        min: 1,
        max: 20,
        step: 1,
        default: 2,
        description: 'Number of components to keep'
      },
      {
        name: 'svd_solver',
        type: 'select',
        options: ['auto', 'full', 'arpack', 'randomized'],
        default: 'auto',
        description: 'Solver to use for SVD computation'
      }
    ],
    
    useCases: [
      {
        title: 'Gene Expression Analysis',
        description: 'Reducing thousands of gene expressions to a manageable number of components for analysis.',
        dataset: 'Microarray Data'
      },
      {
        title: 'Facial Recognition',
        description: 'Eigenfaces approach using PCA to represent faces in a lower-dimensional space.',
        dataset: 'Olivetti Faces'
      }
    ],
    
    comparisons: [
      {
        algorithm: 'LDA',
        comparison: 'PCA is unsupervised and maximizes variance, while LDA is supervised and maximizes class separation'
      },
      {
        algorithm: 't-SNE',
        comparison: 'PCA preserves global structure linearly, while t-SNE preserves local structure non-linearly'
      }
    ],
    
    quiz: [
      {
        question: 'What does PCA primarily seek to maximize?',
        options: [
          'Class separation',
          'Feature correlation',
          'Variance explained',
          'Data compression'
        ],
        correct: 2,
        explanation: 'PCA finds directions (principal components) that maximize the variance in the data.'
      },
      {
        question: 'Why is data standardization important before applying PCA?',
        options: [
          'To make the algorithm run faster',
          'To prevent features with large scales from dominating',
          'To ensure all features are normally distributed',
          'To reduce the number of components needed'
        ],
        correct: 1,
        explanation: 'Without standardization, features with larger scales would dominate the principal components, regardless of their actual importance.'
      }
    ],
    
    projects: [
      {
        title: 'Wine Quality Analysis',
        description: 'Use PCA to reduce the dimensionality of wine chemical measurements and visualize patterns.',
        steps: [
          'Load and explore the Wine Quality dataset',
          'Standardize the features',
          'Apply PCA and determine optimal component count',
          'Visualize wines in 2D/3D space colored by quality',
          'Interpret component loadings'
        ],
        difficulty: 'intermediate',
        xp: 300
      }
    ]
  },

  // DBSCAN AlgoData
  {
    id: 'dbscan',
    title: 'DBSCAN',
    category: 'unsupervised',
    difficulty: 'advanced',
    tags: ['Clustering', 'Density-Based', 'Unsupervised'],
    description: 'Density-Based Spatial Clustering of Applications with Noise groups together points that are closely packed while marking points in low-density regions as outliers.',
    icon: 'object-group',
    lastUpdated: '2023-08-15',
    popularity: 0.78,
    
    concept: {
      overview: 'DBSCAN is a density-based clustering algorithm that groups points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions.',
      analogy: 'Imagine people gathering in a park. DBSCAN would identify groups of people standing close together as clusters, while individuals standing far from any group would be considered noise.',
      history: 'Proposed by Martin Ester, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu in 1996. Won the SIGKDD Test of Time Award in 2014.',
      mathematicalFormulation: {
        coreConcepts: [
          {
            concept: 'Epsilon-neighborhood (ε)',
            description: 'The radius within which to search for neighbors'
          },
          {
            concept: 'MinPts',
            description: 'Minimum number of points required to form a dense region'
          },
          {
            concept: 'Core point',
            description: 'A point that has at least MinPts points within its ε-neighborhood'
          },
          {
            concept: 'Border point',
            description: 'A point that has fewer than MinPts but is reachable from a core point'
          },
          {
            concept: 'Noise point',
            description: 'A point that is not a core point and cannot be reached from any core point'
          }
        ],
        algorithmSteps: [
          'Find all points within ε distance of each point',
          'Identify core points with more than MinPts neighbors',
          'Form clusters from core points and connected border points',
          'Mark remaining points as noise'
        ]
      },
      assumptions: [
        'Clusters are dense regions in data space separated by low-density regions',
        'The density within clusters is roughly uniform',
        'The algorithm can discover clusters of arbitrary shape'
      ]
    },
    
    visualization: {
      visualizerKey: 'dbscan',
      defaultType: 'default',
      description: 'Interactive visualization of DBSCAN clustering algorithm. Observe how the algorithm identifies clusters based on density and separates noise points.',
      instructions: [
        'Adjust epsilon to change the neighborhood radius',
        'Modify MinPts to control the minimum cluster size',
        'Toggle between different data distributions',
        'Enable step-by-step mode to see the clustering process'
      ],
      showTypeSelector: true,
      types: [
        { 
          value: 'default', 
          label: 'Standard View', 
          description: 'Shows final clusters with different colors and noise points',
          default: true 
        },
        { 
          value: 'step-by-step', 
          label: 'Step-by-Step',
          description: 'Visualize the clustering process step by step'
        },
        { 
          value: 'epsilon-circles', 
          label: 'Epsilon Neighborhoods',
          description: 'Show epsilon circles around each point'
        },
        { 
          value: 'core-points', 
          label: 'Core Points Highlight',
          description: 'Highlight core points in clusters'
        },
        { 
          value: 'all', 
          label: 'Complete View',
          description: 'Show all visualization elements together'
        }
      ],
      parameters: {
        n_samples: 150,
        n_clusters: 3,
        cluster_std: 0.5,
        epsilon: 0.3,
        min_samples: 5,
        distribution: 'blobs',
        noise_level: 0.05,
        show_epsilon: false,
        show_core_points: false,
        show_step_by_step: false,
        animation_duration: 2000,
        interactive: true
      },
      performanceTips: [
        'Higher epsilon values will create fewer, larger clusters',
        'Higher MinPts values will create more noise points',
        'Moon and circle distributions show DBSCAN\'s ability to find non-spherical clusters'
      ]
    },
    
    implementations: {
      python: {
        version: "3.9",
        libraries: ['scikit-learn@1.0.2', 'numpy@1.21.5'],
        code: `# Python implementation using scikit-learn
  from sklearn.cluster import DBSCAN
  from sklearn.datasets import make_moons
  import numpy as np

  # Generate sample data
  X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

  # Create and fit model
  dbscan = DBSCAN(eps=0.3, min_samples=5)
  clusters = dbscan.fit_predict(X)

  # Analyze results
  n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
  n_noise = list(clusters).count(-1)

  print(f"Estimated number of clusters: {n_clusters}")
  print(f"Estimated number of noise points: {n_noise}")
  print(f"Cluster labels: {np.unique(clusters)}")`,
        timeComplexity: "O(n log n) with spatial indexing, O(n²) without",
        spaceComplexity: "O(n)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For large datasets, consider using the OPTICS algorithm which is a generalization of DBSCAN. Feature scaling is important for good results."
      },
      r: {
        version: "4.2",
        libraries: ['dbscan@1.1-11', 'fpc@2.2-9'],
        code: `# R implementation using dbscan package
  library(dbscan)

  # Generate sample data
  set.seed(42)
  data <- matrix(c(rnorm(100, mean = 0, sd = 0.3), 
                  rnorm(100, mean = 0, sd = 0.3)), ncol = 2)
  data <- rbind(data, 
                matrix(c(rnorm(100, mean = 3, sd = 0.3), 
                        rnorm(100, mean = 3, sd = 0.3)), ncol = 2))

  # Perform DBSCAN clustering
  dbscan_result <- dbscan(data, eps = 0.3, minPts = 5)

  # Print results
  print(dbscan_result)

  # Plot clusters
  plot(data, col = dbscan_result$cluster + 1L, pch = 20)
  points(data[dbscan_result$cluster == 0, ], col = "black", pch = 4)`,
        timeComplexity: "O(n log n)",
        spaceComplexity: "O(n)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "The dbscan package provides optimized implementations. For very large datasets, use the frNN function for fast nearest neighbor search first."
      },
      javascript: {
        version: "1.7.0",
        libraries: ['ml.js@0.12.0'],
        code: `// JavaScript implementation using ml.js
  import { DBSCAN } from 'ml-clustering';

  // Sample data
  const dataset = [
    [1, 1], [0, 1], [1, 0],
    [10, 10], [10, 13], [13, 13],
    [30, 30], [30, 32], [32, 32], [100, 100]
  ];

  // Create and run DBSCAN
  const dbscan = new DBSCAN();
  const clusters = dbscan.run(dataset, 5, 2);

  // Output results
  console.log('Clusters:', clusters.clusters);
  console.log('Noise:', clusters.noise);

  // The clusters array contains indices of points in each cluster
  clusters.clusters.forEach((cluster, idx) => {
    console.log(\`Cluster \${idx} has \${cluster.length} points\`);
  });`,
        timeComplexity: "O(n²)",
        spaceComplexity: "O(n²)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For better performance with large datasets in JavaScript, consider implementing with a spatial index like k-d tree. This implementation is suitable for small to medium datasets."
      },
      cpp: {
        version: "17",
        libraries: ['mlpack@3.4.2'],
        code: `// C++ implementation using mlpack
  #include <mlpack.hpp>
  #include <iostream>

  using namespace mlpack;
  using namespace arma;

  int main() {
    // Sample data
    mat data = {{1, 1}, {0, 1}, {1, 0},
                {10, 10}, {10, 13}, {13, 13},
                {30, 30}, {30, 32}, {32, 32}, {100, 100}};

    // The DBSCAN object requires the epsilon and minimum points parameters
    const double epsilon = 5.0;
    const size_t minPoints = 2;

    // Perform DBSCAN clustering
    arma::Row<size_t> assignments;
    dbscan::DBSCAN<> dbscan(epsilon, minPoints);
    dbscan.Cluster(data, assignments);

    // Output results
    std::cout << "Cluster assignments:" << std::endl;
    for (size_t i = 0; i < assignments.n_elem; ++i) {
      std::cout << "Point " << i << " -> Cluster " << assignments[i] << std::endl;
    }

    return 0;
  }`,
        timeComplexity: "O(n log n) with spatial indexing",
        spaceComplexity: "O(n)",
        author: {
          name: "Numerical Computing Team"
        },
        lastUpdated: "2023-03-05",
        notes: "mlpack's DBSCAN implementation uses range trees for efficient neighbor searches. For very high-dimensional data, consider approximate nearest neighbor methods."
      }
    },

    prosCons: {
      strengths: [
        'Does not require specifying number of clusters',
        'Can find clusters of arbitrary shape',
        'Robust to outliers',
        'Has a notion of noise'
      ],
      weaknesses: [
        'Struggles with clusters of varying densities',
        'Sensitive to parameters ε and MinPts',
        'Not entirely deterministic on border points',
        'Performance degrades with high-dimensional data'
      ]
    },
    
    hyperparameters: [
      {
        name: 'eps',
        type: 'range',
        min: 0.01,
        max: 2.0,
        step: 0.01,
        default: 0.5,
        description: 'The maximum distance between two samples for one to be considered as in the neighborhood of the other'
      },
      {
        name: 'min_samples',
        type: 'range',
        min: 1,
        max: 20,
        step: 1,
        default: 5,
        description: 'The number of samples in a neighborhood for a point to be considered as a core point'
      }
    ],
    
    useCases: [
      {
        title: 'Anomaly Detection',
        description: 'Identifying unusual patterns in network traffic that do not conform to expected behavior.',
        dataset: 'Network Intrusion Detection'
      },
      {
        title: 'Spatial Data Analysis',
        description: 'Grouping geographical locations based on density for urban planning applications.',
        dataset: 'Urban Points of Interest'
      }
    ],
    
    comparisons: [
      {
        algorithm: 'K-Means',
        comparison: 'DBSCAN can find arbitrarily shaped clusters and identify noise, while K-Means assumes spherical clusters and requires specifying K'
      },
      {
        algorithm: 'Hierarchical Clustering',
        comparison: 'DBSCAN is more efficient for large datasets and automatically determines the number of clusters'
      }
    ],
    
    quiz: [
      {
        question: 'What is the main advantage of DBSCAN over K-Means?',
        options: [
          'Faster computation time',
          'Ability to find non-spherical clusters',
          'Better performance with high-dimensional data',
          'No parameters to tune'
        ],
        correct: 1,
        explanation: 'DBSCAN can find clusters of arbitrary shape, while K-Means assumes spherical clusters.'
      },
      {
        question: 'Which of the following is NOT a type of point in DBSCAN?',
        options: [
          'Core point',
          'Border point',
          'Noise point',
          'Centroid point'
        ],
        correct: 3,
        explanation: 'Centroid point is a concept in K-Means, not DBSCAN. DBSCAN has core, border, and noise points.'
      }
    ],
    
    projects: [
      {
        title: 'Customer Segmentation',
        description: 'Use DBSCAN to identify customer groups based on purchasing behavior for a retail company.',
        steps: [
          'Load and preprocess customer transaction data',
          'Feature engineering to create meaningful customer attributes',
          'Apply DBSCAN to identify customer segments',
          'Analyze and interpret the clusters',
          'Compare with K-Means results'
        ],
        difficulty: 'intermediate',
        xp: 350
      }
    ]
  },

  // Bagging AlgoData
  {
    id: 'bagging',
    title: 'Bagging (Bootstrap Aggregating)',
    category: 'ensemble',
    difficulty: 'beginner',
    tags: ['Ensemble', 'Bootstrap', 'Supervised'],
    description: 'An ensemble method that reduces variance by training multiple models on different subsets of the data and averaging their predictions.',
    icon: 'cubes',
    lastUpdated: '2023-08-15',
    popularity: 0.78,
    
    concept: {
      overview: 'Bagging creates multiple versions of a predictor and aggregates them to get an improved predictor. It works particularly well for high-variance, low-bias models like decision trees.',
      analogy: 'Like asking multiple doctors for a diagnosis and taking the majority opinion - each doctor sees a slightly different set of symptoms but their collective wisdom is more reliable.',
      history: 'Developed by Leo Breiman in 1996 as a way to improve the stability and accuracy of machine learning algorithms.',
      mathematicalFormulation: {
        equation: 'ŷ = (1/B) * Σ(ŷ_b) for regression, ŷ = argmax(Σ(I(ŷ_b = c))) for classification',
        variables: [
          { symbol: 'B', description: 'Number of bootstrap samples' },
          { symbol: 'ŷ_b', description: 'Prediction from b-th bootstrap model' },
          { symbol: 'I()', description: 'Indicator function' }
        ],
        bootstrapProcess: 'Each model is trained on a random sample with replacement from the original dataset',
        aggregationMethods: [
          'Averaging for regression problems',
          'Majority voting for classification problems'
        ]
      },
      assumptions: [
        'Base learners should be unstable (high variance)',
        'Base learners should have low bias',
        'Models should be diverse (decorrelated errors)'
      ]
    },
    
    visualization: {
      visualizerKey: 'bagging',
      defaultType: 'bootstrap-process',
      description: 'Interactive visualization of bagging algorithm. Observe how multiple models trained on different data subsets combine to form a more robust predictor.',
      instructions: [
        'Adjust number of estimators to see ensemble size effect',
        'Modify base learner complexity',
        'Toggle between showing individual models vs ensemble',
        'Observe how variance reduces as ensemble size increases'
      ],
      showTypeSelector: true,
      types: [
        { 
          value: 'bootstrap-process', 
          label: 'Bootstrap Sampling', 
          description: 'Visualize how different data subsets are created',
          default: true 
        },
        { 
          value: 'individual-models', 
          label: 'Individual Models',
          description: 'Show predictions from each base learner'
        },
        { 
          value: 'ensemble-result', 
          label: 'Ensemble Prediction',
          description: 'Final aggregated prediction'
        },
        { 
          value: 'variance-reduction', 
          label: 'Variance Analysis',
          description: 'Visualize how variance decreases with ensemble size'
        },
        { 
          value: 'all', 
          label: 'Complete View',
          description: 'Show all bagging components together'
        }
      ],
      parameters: {
        n_samples: 100,
        n_estimators: 10,
        base_learner: 'decision-tree',
        max_depth: 3,
        bootstrap_ratio: 0.8,
        noise: 0.3,
        show_bootstraps: true,
        show_individuals: false,
        show_ensemble: true,
        show_variance: false,
        animation_duration: 1800,
        interactive: true
      },
      performanceTips: [
        'More estimators reduce variance but increase computation time',
        'Decision trees work well as base learners due to high variance',
        'Bootstrap ratio of 0.6-0.8 typically works well'
      ]
    },
    
    implementations: {
      python: {
        version: "3.9",
        libraries: ['scikit-learn@1.0.2', 'numpy@1.21.5'],
        code: `# Python implementation using scikit-learn
  from sklearn.ensemble import BaggingClassifier, BaggingRegressor
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.model_selection import train_test_split
  from sklearn.datasets import make_classification
  import numpy as np

  # Generate sample data
  X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                            n_redundant=5, random_state=42)

  # Split data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Create base estimator
  base_estimator = DecisionTreeClassifier(max_depth=4)

  # Create and fit bagging classifier
  bagging = BaggingClassifier(
      estimator=base_estimator,
      n_estimators=50,
      max_samples=0.8,
      bootstrap=True,
      random_state=42
  )
  bagging.fit(X_train, y_train)

  # Evaluate
  print(f"Training accuracy: {bagging.score(X_train, y_train):.3f}")
  print(f"Test accuracy: {bagging.score(X_test, y_test):.3f}")
  print(f"Number of estimators: {len(bagging.estimators_)}")`,
        timeComplexity: "O(B * T) where B is number of estimators and T is base learner time",
        spaceComplexity: "O(B * S) where S is base learner space",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For regression problems, use BaggingRegressor. Can use any base estimator, but decision trees work best."
      },
      r: {
        version: "4.2",
        libraries: ['ipred@0.9-13', 'randomForest@4.7-1.1'],
        code: `# R implementation using ipred package
  library(ipred)
  library(caret)

  # Generate sample data
  set.seed(42)
  data <- twoClassSim(1000)

  # Split data
  train_index <- createDataPartition(data$Class, p=0.8, list=FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]

  # Create bagging model
  bagging_model <- bagging(
    Class ~ .,
    data = train_data,
    nbagg = 50,
    coob = TRUE
  )

  # Evaluate
  predictions <- predict(bagging_model, test_data)
  confusion_matrix <- confusionMatrix(predictions, test_data$Class)
  print(confusion_matrix)`,
        timeComplexity: "O(B * T)",
        spaceComplexity: "O(B * S)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "The ipred package provides general bagging. For specifically bagged decision trees, use randomForest with mtry = number of features."
      },
      javascript: {
        version: "1.7.0",
        libraries: ['ml.js@0.12.0'],
        code: `// JavaScript implementation using ml.js
  import { Bagging } from 'ml-ensemble';

  // Sample training data
  const features = [
    [0, 0], [0, 1], [1, 0], [1, 1],
    [0.5, 0.5], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]
  ];
  const labels = [0, 1, 1, 1, 0, 1, 1, 1];

  // Create bagging classifier
  const options = {
    base: {
      name: 'DecisionTree',
      options: { maxDepth: 3 }
    },
    numModels: 10,
    sampleSize: 0.7
  };

  const classifier = new Bagging(options);
  classifier.train(features, labels);

  // Predict
  const test = [[0.6, 0.4]];
  const prediction = classifier.predict(test);
  console.log('Prediction:', prediction);

  // Get individual model predictions
  const allPredictions = classifier.predictEach(test);
  console.log('All predictions:', allPredictions);`,
        timeComplexity: "O(B * T)",
        spaceComplexity: "O(B * S)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "ml.js implementation is suitable for smaller datasets. For larger datasets, consider server-side processing."
      },
      cpp: {
        version: "17",
        libraries: ['mlpack@3.4.2'],
        code: `// C++ implementation using mlpack
  #include <mlpack.hpp>
  #include <iostream>

  using namespace mlpack;
  using namespace arma;

  int main() {
    // Sample data
    mat features = randu<mat>(20, 1000); // 20 features, 1000 samples
    Row<size_t> labels = randi<Row<size_t>>(1000, distr_param(0, 1));

    // Create bagging model (using decision tree as base)
    Bagging<> bagging(
      50, // number of estimators
      0.8, // bootstrap ratio
      DecisionTree<>() // base estimator
    );

    // Train model
    bagging.Train(features, labels);

    // Predict
    mat testPoint = randu<mat>(20, 1);
    size_t prediction;
    bagging.Classify(testPoint, prediction);
    std::cout << "Prediction: " << prediction << std::endl;

    return 0;
  }`,
        timeComplexity: "O(B * T)",
        spaceComplexity: "O(B * S)",
        author: {
          name: "Numerical Computing Team"
        },
        lastUpdated: "2023-03-05",
        notes: "mlpack provides flexible bagging implementation with various base estimators. For parallel training, enable OpenMP support."
      }
    },

    prosCons: {
      strengths: [
        'Reduces variance and helps avoid overfitting',
        'Can be parallelized easily',
        'Works with any base estimator',
        'Improves stability of unstable algorithms'
      ],
      weaknesses: [
        'Increased computational cost',
        'Less interpretable than single models',
        'May not help with biased base estimators',
        'Can be memory intensive'
      ]
    },
    
    hyperparameters: [
      {
        name: 'n_estimators',
        type: 'range',
        min: 10,
        max: 200,
        step: 10,
        default: 50,
        description: 'Number of base estimators in the ensemble'
      },
      {
        name: 'max_samples',
        type: 'range',
        min: 0.1,
        max: 1.0,
        step: 0.1,
        default: 0.8,
        description: 'Proportion of samples to draw from X to train each base estimator'
      },
      {
        name: 'bootstrap',
        type: 'boolean',
        default: true,
        description: 'Whether samples are drawn with replacement'
      }
    ],
    
    useCases: [
      {
        title: 'Financial Risk Prediction',
        description: 'Predicting credit default risk with improved stability using bagged decision trees.',
        dataset: 'German Credit Risk'
      },
      {
        title: 'Medical Diagnosis',
        description: 'Improving diagnostic accuracy by combining predictions from multiple models trained on different patient subsets.',
        dataset: 'Breast Cancer Wisconsin'
      }
    ],
    
    comparisons: [
      {
        algorithm: 'Single Decision Tree',
        comparison: 'Bagging reduces variance and overfitting compared to a single tree, at the cost of interpretability'
      },
      {
        algorithm: 'Boosting',
        comparison: 'Bagging builds models in parallel while boosting builds sequentially; bagging reduces variance, boosting reduces bias'
      }
    ],
    
    quiz: [
      {
        question: 'What is the primary goal of bagging?',
        options: [
          'To reduce bias in the model',
          'To reduce variance in the model',
          'To make the model more interpretable',
          'To reduce training time'
        ],
        correct: 1,
        explanation: 'Bagging primarily reduces variance by averaging multiple models trained on different data subsets.'
      },
      {
        question: 'Which type of base learners work best with bagging?',
        options: [
          'High bias, low variance learners',
          'Low bias, high variance learners',
          'Linear models',
          'Neural networks with many layers'
        ],
        correct: 1,
        explanation: 'Bagging is most effective with unstable, high-variance learners like decision trees.'
      }
    ],
    
    projects: [
      {
        title: 'Bagged Decision Trees for Classification',
        description: 'Implement a bagging ensemble with decision trees to improve classification performance.',
        steps: [
          'Load a classification dataset',
          'Train a single decision tree as baseline',
          'Implement bagging with multiple decision trees',
          'Compare performance with single tree',
          'Analyze how ensemble size affects performance'
        ],
        difficulty: 'intermediate',
        xp: 350
      }
    ]
  },

  // AdaBoost AlgoData
  {
    id: 'adaboost',
    title: 'AdaBoost (Adaptive Boosting)',
    category: 'ensemble',
    difficulty: 'intermediate',
    tags: ['Ensemble', 'Boosting', 'Supervised'],
    description: 'An iterative ensemble method that combines weak learners, focusing more on difficult examples in each iteration.',
    icon: 'rocket',
    lastUpdated: '2023-09-05',
    popularity: 0.82,
    
    concept: {
      overview: 'AdaBoost works by sequentially applying a weak classifier to repeatedly modified versions of the data. The algorithm increases the weights of misclassified instances so subsequent classifiers focus more on difficult cases.',
      analogy: 'Like a student who focuses more on the topics they struggle with - each study session targets the areas where they made mistakes previously.',
      history: 'Developed by Yoav Freund and Robert Schapire in 1996, won the Gödel Prize in 2003 for its theoretical significance.',
      mathematicalFormulation: {
        algorithmSteps: [
          'Initialize equal weights for all training instances',
          'For each iteration: train weak learner on weighted data',
          'Calculate error and update instance weights',
          'Calculate classifier weight based on performance',
          'Combine weak classifiers into strong classifier'
        ],
        weightUpdate: 'w_i = w_i * exp(α * I(y_i ≠ ŷ_i)) where α = 0.5 * ln((1-error)/error)',
        finalPrediction: 'ŷ = sign(Σ(α_t * h_t(x)))'
      },
      assumptions: [
        'Weak learners should perform better than random guessing',
        'Training data should be representative',
        'Weak learners should be diverse'
      ]
    },
    
    visualization: {
      visualizerKey: 'adaboost',
      defaultType: 'weight-evolution',
      description: 'Interactive visualization of AdaBoost algorithm. Observe how instance weights change and how weak learners combine to form a strong classifier.',
      instructions: [
        'Adjust number of iterations to see boosting process',
        'Observe how instance weights change for misclassified points',
        'Watch how decision boundary evolves with each weak learner',
        'Toggle between showing individual weak learners vs final ensemble'
      ],
      showTypeSelector: true,
      types: [
        { 
          value: 'weight-evolution', 
          label: 'Weight Evolution', 
          description: 'Visualize how instance weights change across iterations',
          default: true 
        },
        { 
          value: 'weak-learners', 
          label: 'Weak Learners',
          description: 'Show individual weak learner decisions'
        },
        { 
          value: 'ensemble-evolution', 
          label: 'Ensemble Progress',
          description: 'How the combined classifier improves over iterations'
        },
        { 
          value: 'error-convergence', 
          label: 'Error Convergence',
          description: 'Training and test error across iterations'
        },
        { 
          value: 'all', 
          label: 'Complete View',
          description: 'All AdaBoost components together'
        }
      ],
      parameters: {
        n_samples: 100,
        n_estimators: 10,
        base_learner: 'decision-stump',
        learning_rate: 1.0,
        noise: 0.3,
        show_weights: true,
        show_weak_learners: false,
        show_ensemble: true,
        show_error: false,
        animation_duration: 2000,
        interactive: true
      },
      performanceTips: [
        'Decision stumps work well as weak learners',
        'Lower learning rates require more iterations but can achieve better performance',
        'Watch how difficult examples get higher weights across iterations'
      ]
    },
    
    implementations: {
      python: {
        version: "3.9",
        libraries: ['scikit-learn@1.0.2', 'numpy@1.21.5'],
        code: `# Python implementation using scikit-learn
  from sklearn.ensemble import AdaBoostClassifier
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.model_selection import train_test_split
  from sklearn.datasets import make_classification
  from sklearn.metrics import accuracy_score
  import numpy as np

  # Generate sample data
  X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                            n_redundant=5, random_state=42)

  # Split data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Create base estimator (typically decision stumps)
  base_estimator = DecisionTreeClassifier(max_depth=1)

  # Create and fit AdaBoost classifier
  adaboost = AdaBoostClassifier(
      estimator=base_estimator,
      n_estimators=50,
      learning_rate=1.0,
      random_state=42
  )
  adaboost.fit(X_train, y_train)

  # Evaluate
  train_pred = adaboost.predict(X_train)
  test_pred = adaboost.predict(X_test)

  print(f"Training accuracy: {accuracy_score(y_train, train_pred):.3f}")
  print(f"Test accuracy: {accuracy_score(y_test, test_pred):.3f}")
  print(f"Number of estimators: {len(adaboost.estimators_)}")
  print(f"Estimator weights: {adaboost.estimator_weights_}")`,
        timeComplexity: "O(T * B) where T is number of iterations and B is base learner time",
        spaceComplexity: "O(T * S) where S is base learner space",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For regression problems, use AdaBoostRegressor. Decision stumps (max_depth=1) are commonly used as weak learners."
      },
      r: {
        version: "4.2",
        libraries: ['adabag@4.2', 'rpart@4.1.16'],
        code: `# R implementation using adabag package
  library(adabag)
  library(caret)

  # Generate sample data
  set.seed(42)
  data <- twoClassSim(1000)

  # Split data
  train_index <- createDataPartition(data$Class, p=0.8, list=FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]

  # Create AdaBoost model
  adaboost_model <- boosting(
    Class ~ .,
    data = train_data,
    boos = TRUE,
    mfinal = 50,
    coeflearn = 'Breiman'
  )

  # Evaluate
  predictions <- predict(adaboost_model, test_data)
  confusion_matrix <- confusionMatrix(predictions$class, test_data$Class)
  print(confusion_matrix)

  # View importance
  importance <- adaboost_model$importance
  print(importance)`,
        timeComplexity: "O(T * B)",
        spaceComplexity: "O(T * S)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "The adabag package provides comprehensive boosting implementation. Different coeflearn methods affect how learner weights are calculated."
      },
      javascript: {
        version: "1.7.0",
        libraries: ['ml.js@0.12.0'],
        code: `// JavaScript implementation using ml.js
  import { AdaBoost } from 'ml-ensemble';

  // Sample training data
  const features = [
    [0, 0], [0, 1], [1, 0], [1, 1],
    [0.5, 0.5], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]
  ];
  const labels = [0, 1, 1, 1, 0, 1, 1, 1];

  // Create AdaBoost classifier
  const options = {
    base: {
      name: 'DecisionStump'
    },
    numIterations: 10,
    learningRate: 1.0
  };

  const classifier = new AdaBoost(options);
  classifier.train(features, labels);

  // Predict
  const test = [[0.6, 0.4]];
  const prediction = classifier.predict(test);
  console.log('Prediction:', prediction);

  // Get iteration details
  const iterations = classifier.getIterations();
  console.log('Iteration details:', iterations);`,
        timeComplexity: "O(T * B)",
        spaceComplexity: "O(T * S)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "ml.js implementation is suitable for educational purposes and smaller datasets."
      },
      cpp: {
        version: "17",
        libraries: ['mlpack@3.4.2'],
        code: `// C++ implementation using mlpack
  #include <mlpack.hpp>
  #include <iostream>

  using namespace mlpack;
  using namespace arma;

  int main() {
    // Sample data
    mat features = randu<mat>(20, 1000); // 20 features, 1000 samples
    Row<size_t> labels = randi<Row<size_t>>(1000, distr_param(0, 1));

    // Create AdaBoost model
    AdaBoost<> adaboost(
      50, // number of iterations
      DecisionStump<>() // weak learner
    );

    // Train model
    adaboost.Train(features, labels);

    // Predict
    mat testPoint = randu<mat>(20, 1);
    size_t prediction;
    adaboost.Classify(testPoint, prediction);
    std::cout << "Prediction: " << prediction << std::endl;

    return 0;
  }`,
        timeComplexity: "O(T * B)",
        spaceComplexity: "O(T * S)",
        author: {
          name: "Numerical Computing Team"
        },
        lastUpdated: "2023-03-05",
        notes: "mlpack provides efficient AdaBoost implementation with various weak learner options. Decision stumps are most commonly used."
      }
    },

    prosCons: {
      strengths: [
        'Can achieve high accuracy with weak learners',
        'Less prone to overfitting than some algorithms',
        'Adapts to hard examples',
        'No need for extensive parameter tuning'
      ],
      weaknesses: [
        'Sensitive to noisy data and outliers',
        'Weak learners must be better than random guessing',
        'Sequential training cannot be parallelized',
        'Performance can degrade with too many iterations'
      ]
    },
    
    hyperparameters: [
      {
        name: 'n_estimators',
        type: 'range',
        min: 10,
        max: 200,
        step: 10,
        default: 50,
        description: 'Maximum number of estimators at which boosting is terminated'
      },
      {
        name: 'learning_rate',
        type: 'range',
        min: 0.01,
        max: 2.0,
        step: 0.01,
        default: 1.0,
        description: 'Weight applied to each classifier at each boosting iteration'
      },
      {
        name: 'base_estimator',
        type: 'select',
        options: ['decision-stump', 'shallow-tree', 'other'],
        default: 'decision-stump',
        description: 'The base estimator from which the boosted ensemble is built'
      }
    ],
    
    useCases: [
      {
        title: 'Face Detection',
        description: 'One of the first successful applications of AdaBoost was in the Viola-Jones face detection algorithm.',
        dataset: 'Viola-Jones Face Detection'
      },
      {
        title: 'Text Classification',
        description: 'Improving spam detection by focusing on difficult-to-classify emails across iterations.',
        dataset: 'Spambase'
      }
    ],
    
    comparisons: [
      {
        algorithm: 'Bagging',
        comparison: 'AdaBoost builds models sequentially focusing on errors, while bagging builds models in parallel on different data subsets'
      },
      {
        algorithm: 'Gradient Boosting',
        comparison: 'AdaBoost uses exponential loss and instance reweighting, while gradient boosting uses gradient descent and residual fitting'
      }
    ],
    
    quiz: [
      {
        question: 'What does AdaBoost stand for?',
        options: [
          'Adaptive Bootstrap',
          'Advanced Boosting',
          'Adaptive Boosting',
          'Algorithmic Boosting'
        ],
        correct: 2,
        explanation: 'AdaBoost stands for Adaptive Boosting, as it adapts to the errors of previous classifiers.'
      },
      {
        question: 'How does AdaBoost handle difficult examples?',
        options: [
          'By ignoring them to focus on easier patterns',
          'By decreasing their weight in subsequent iterations',
          'By increasing their weight in subsequent iterations',
          'By using a different algorithm for difficult cases'
        ],
        correct: 2,
        explanation: 'AdaBoost increases the weights of misclassified instances so subsequent classifiers focus more on them.'
      }
    ],
    
    projects: [
      {
        title: 'AdaBoost for Binary Classification',
        description: 'Implement AdaBoost with decision stumps to solve a binary classification problem.',
        steps: [
          'Load a binary classification dataset',
          'Implement AdaBoost algorithm from scratch',
          'Visualize how instance weights change across iterations',
          'Compare with other ensemble methods',
          'Analyze the effect of learning rate and number of estimators'
        ],
        difficulty: 'intermediate',
        xp: 400
      }
    ]
  },

  // GBM AlgoData
  {
    id: 'gbm',
    title: 'Gradient Boosting Machines (GBM)',
    category: 'ensemble',
    difficulty: 'intermediate',
    tags: ['Ensemble', 'Boosting', 'Supervised'],
    description: 'A powerful boosting algorithm that builds models sequentially, with each new model correcting the errors of the previous ones using gradient descent.',
    icon: 'layer-group',
    lastUpdated: '2023-10-12',
    popularity: 0.92,
    
    concept: {
      overview: 'GBM builds models in a stage-wise fashion like other boosting methods, but generalizes them by allowing optimization of an arbitrary differentiable loss function.',
      analogy: 'Like a golfer adjusting their swing - each adjustment corrects for the error of the previous swing, gradually getting closer to the hole.',
      history: 'Introduced by Jerome Friedman in 2001 as a generalization of boosting algorithms to statistical gradient descent.',
      mathematicalFormulation: {
        algorithmSteps: [
          'Initialize model with constant value',
          'For each iteration: compute pseudo-residuals (negative gradient)',
          'Fit weak learner to pseudo-residuals',
          'Compute optimal step size',
          'Update model with new weak learner'
        ],
        gradientCalculation: 'r_{im} = -[∂L(y_i, F(x_i))/∂F(x_i)]_{F(x)=F_{m-1}(x)}',
        modelUpdate: 'F_m(x) = F_{m-1}(x) + γ_m * h_m(x) where γ_m is step size'
      },
      assumptions: [
        'Loss function should be differentiable',
        'Weak learners should be capable of fitting residuals',
        'Training data should be representative of the population'
      ]
    },
    
    visualization: {
      visualizerKey: 'gbm',
      defaultType: 'residual-fitting',
      description: 'Interactive visualization of gradient boosting. Observe how each new model fits the residuals of the previous ensemble.',
      instructions: [
        'Adjust number of boosting iterations',
        'Observe how residuals are calculated and fitted',
        'Watch the ensemble prediction improve gradually',
        'Toggle between regression and classification modes'
      ],
      showTypeSelector: true,
      types: [
        { 
          value: 'residual-fitting', 
          label: 'Residual Fitting', 
          description: 'Visualize how each new model fits the residuals of previous ones',
          default: true 
        },
        { 
          value: 'prediction-evolution', 
          label: 'Prediction Evolution',
          description: 'How the ensemble prediction improves across iterations'
        },
        { 
          value: 'loss-reduction', 
          label: 'Loss Reduction',
          description: 'Training and validation loss across iterations'
        },
        { 
          value: 'feature-importance', 
          label: 'Feature Importance',
          description: 'Feature importance based on usage in trees'
        },
        { 
          value: 'all', 
          label: 'Complete View',
          description: 'All GBM components together'
        }
      ],
      parameters: {
        n_samples: 100,
        n_estimators: 10,
        learning_rate: 0.1,
        max_depth: 3,
        loss_function: 'squared-error',
        subsample: 1.0,
        show_residuals: true,
        show_predictions: true,
        show_loss: false,
        show_importance: false,
        animation_duration: 2200,
        interactive: true
      },
      performanceTips: [
        'Lower learning rates with more iterations often yield better results',
        'Subsampling (stochastic gradient boosting) can improve generalization',
        'Monitor validation loss to prevent overfitting'
      ]
    },
    
    implementations: {
      python: {
        version: "3.9",
        libraries: ['scikit-learn@1.0.2', 'xgboost@1.6.1', 'lightgbm@3.3.2'],
        code: `# Python implementation using scikit-learn
  from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
  from sklearn.model_selection import train_test_split
  from sklearn.datasets import make_classification
  from sklearn.metrics import accuracy_score, mean_squared_error
  import numpy as np

  # Generate sample data
  X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                            n_redundant=5, random_state=42)

  # Split data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Create and fit Gradient Boosting classifier
  gbm = GradientBoostingClassifier(
      n_estimators=100,
      learning_rate=0.1,
      max_depth=3,
      subsample=0.8,
      random_state=42
  )
  gbm.fit(X_train, y_train)

  # Evaluate
  train_pred = gbm.predict(X_train)
  test_pred = gbm.predict(X_test)

  print(f"Training accuracy: {accuracy_score(y_train, train_pred):.3f}")
  print(f"Test accuracy: {accuracy_score(y_test, test_pred):.3f}")
  print(f"Number of estimators: {len(gbm.estimators_)}")
  print(f"Feature importances: {gbm.feature_importances_}")

  # Using XGBoost (often more efficient)
  import xgboost as xgb
  xgb_model = xgb.XGBClassifier(
      n_estimators=100,
      learning_rate=0.1,
      max_depth=3,
      subsample=0.8,
      random_state=42
  )
  xgb_model.fit(X_train, y_train)`,
        timeComplexity: "O(T * B * n) where T is iterations, B is tree building time, n is samples",
        spaceComplexity: "O(T * S) where S is tree space",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For large datasets, consider using XGBoost or LightGBM for better performance and memory efficiency."
      },
      r: {
        version: "4.2",
        libraries: ['gbm@2.1.8.1', 'xgboost@1.6.0.1', 'caret@6.0-93'],
        code: `# R implementation using gbm package
  library(gbm)
  library(caret)

  # Generate sample data
  set.seed(42)
  data <- twoClassSim(1000)

  # Split data
  train_index <- createDataPartition(data$Class, p=0.8, list=FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]

  # Create GBM model
  gbm_model <- gbm(
    Class ~ .,
    data = train_data,
    distribution = "bernoulli",
    n.trees = 100,
    interaction.depth = 3,
    shrinkage = 0.1,
    bag.fraction = 0.8,
    cv.folds = 5,
    verbose = FALSE
  )

  # Find optimal number of trees
  best_iter <- gbm.perf(gbm_model, method = "cv")

  # Predict
  predictions <- predict(gbm_model, test_data, n.trees = best_iter, type = "response")
  predicted_classes <- ifelse(predictions > 0.5, "Class1", "Class2")
  confusion_matrix <- confusionMatrix(factor(predicted_classes), test_data$Class)
  print(confusion_matrix)

  # Feature importance
  importance <- summary(gbm_model, plotit = FALSE)
  print(importance)`,
        timeComplexity: "O(T * B * n)",
        spaceComplexity: "O(T * S)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "The gbm package provides the original implementation. For better performance, use xgboost or lightgbm packages."
      },
      javascript: {
        version: "1.7.0",
        libraries: ['xgboost@1.5.1'],
        code: `// JavaScript implementation using XGBoost.js
  import * as xgboost from 'xgboost';

  // Note: XGBoost.js typically runs in Node.js environment
  // This is a conceptual example

  async function runGBM() {
    // Sample training data
    const features = [
      [0, 0], [0, 1], [1, 0], [1, 1],
      [0.5, 0.5], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]
    ];
    const labels = [0, 1, 1, 1, 0, 1, 1, 1];

    // Create DMatrix (XGBoost data structure)
    const dtrain = new xgboost.DMatrix(features, labels);

    // Set parameters
    const params = {
      objective: 'binary:logistic',
      max_depth: 3,
      learning_rate: 0.1,
      n_estimators: 10,
      subsample: 0.8
    };

    // Train model
    const model = await xgboost.train(params, dtrain, 10);

    // Predict
    const dtest = new xgboost.DMatrix([[0.6, 0.4]]);
    const prediction = await model.predict(dtest);
    console.log('Prediction:', prediction);
  }

  runGBM().catch(console.error);`,
        timeComplexity: "O(T * B * n)",
        spaceComplexity: "O(T * S)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "XGBoost.js requires Node.js environment. For browser-based implementations, consider server-side API calls."
      },
      cpp: {
        version: "17",
        libraries: ['xgboost@1.6.1', 'eigen@3.4.0'],
        code: `// C++ implementation using XGBoost
  #include <xgboost/c_api.h>
  #include <iostream>
  #include <vector>

  int main() {
    // Create sample data
    std::vector<float> data = {0,0, 0,1, 1,0, 1,1, 0.5,0.5, 0.2,0.8, 0.8,0.2, 0.3,0.7};
    std::vector<float> labels = {0, 1, 1, 1, 0, 1, 1, 1};
    
    // Create DMatrix
    DMatrixHandle dtrain;
    XGDMatrixCreateFromMat(data.data(), 8, 2, -1, &dtrain);
    XGDMatrixSetFloatInfo(dtrain, "label", labels.data(), 8);
    
    // Create booster and set parameters
    BoosterHandle booster;
    XGBoosterCreate(&dtrain, 1, &booster);
    
    XGBoosterSetParam(booster, "objective", "binary:logistic");
    XGBoosterSetParam(booster, "max_depth", "3");
    XGBoosterSetParam(booster, "eta", "0.1");
    XGBoosterSetParam(booster, "subsample", "0.8");
    
    // Train for 10 iterations
    for (int i = 0; i < 10; i++) {
      XGBoosterUpdateOneIter(booster, i, dtrain);
    }
    
    // Create test data
    std::vector<float> test_data = {0.6, 0.4};
    DMatrixHandle dtest;
    XGDMatrixCreateFromMat(test_data.data(), 1, 2, -1, &dtest);
    
    // Predict
    bst_ulong out_len;
    const float* out_result;
    XGBoosterPredict(booster, dtest, 0, 0, &out_len, &out_result);
    
    std::cout << "Prediction: " << out_result[0] << std::endl;
    
    // Clean up
    XGDMatrixFree(dtrain);
    XGDMatrixFree(dtest);
    XGBoosterFree(booster);
    
    return 0;
  }`,
        timeComplexity: "O(T * B * n)",
        spaceComplexity: "O(T * S)",
        author: {
          name: "XGBoost Development Team"
        },
        lastUpdated: "2023-03-05",
        notes: "XGBoost C API provides the most efficient implementation. For easier C++ interface, consider the XGBoost4J package."
      }
    },

    prosCons: {
      strengths: [
        'Often achieves state-of-the-art performance on tabular data',
        'Handles mixed data types well',
        'Provides feature importance measures',
        'Flexible with different loss functions'
      ],
      weaknesses: [
        'Can be computationally expensive',
        'Requires careful tuning to avoid overfitting',
        'Sequential training is not easily parallelizable',
        'Less interpretable than single trees'
      ]
    },
    
    hyperparameters: [
      {
        name: 'n_estimators',
        type: 'range',
        min: 10,
        max: 1000,
        step: 10,
        default: 100,
        description: 'Number of boosting stages to perform'
      },
      {
        name: 'learning_rate',
        type: 'range',
        min: 0.01,
        max: 0.3,
        step: 0.01,
        default: 0.1,
        description: 'Shrinks the contribution of each tree to prevent overfitting'
      },
      {
        name: 'max_depth',
        type: 'range',
        min: 1,
        max: 10,
        step: 1,
        default: 3,
        description: 'Maximum depth of the individual regression estimators'
      },
      {
        name: 'subsample',
        type: 'range',
        min: 0.1,
        max: 1.0,
        step: 0.1,
        default: 1.0,
        description: 'Fraction of samples to be used for fitting the individual base learners'
      }
    ],
    
    useCases: [
      {
        title: 'Predictive Maintenance',
        description: 'Predicting equipment failure using sensor data with gradient boosting machines.',
        dataset: 'NASA Turbofan Engine Degradation'
      },
      {
        title: 'Click-Through Rate Prediction',
        description: 'Online advertising CTR prediction using gradient boosted trees.',
        dataset: 'Criteo Display Advertising'
      }
    ],
    
    comparisons: [
      {
        algorithm: 'Random Forest',
        comparison: 'GBM builds trees sequentially to reduce bias, while random forest builds trees in parallel to reduce variance'
      },
      {
        algorithm: 'AdaBoost',
        comparison: 'GBM uses gradient descent to optimize any differentiable loss function, while AdaBoost uses exponential loss and instance reweighting'
      }
    ],
    
    quiz: [
      {
        question: 'What is the key innovation of gradient boosting compared to earlier boosting methods?',
        options: [
          'It uses bootstrap sampling',
          'It can optimize any differentiable loss function',
          'It builds models in parallel',
          'It only works with decision trees'
        ],
        correct: 1,
        explanation: 'The key innovation is that gradient boosting can optimize any differentiable loss function using gradient descent.'
      },
      {
        question: 'How does gradient boosting prevent overfitting?',
        options: [
          'By using shallow trees only',
          'By using a learning rate (shrinkage)',
          'By stopping after the first few iterations',
          'By ignoring noisy data points'
        ],
        correct: 1,
        explanation: 'Gradient boosting uses a learning rate (shrinkage) to reduce the contribution of each tree, which helps prevent overfitting.'
      }
    ],
    
    projects: [
      {
        title: 'Gradient Boosting for Regression',
        description: 'Implement gradient boosting to solve a regression problem and compare it with other regression techniques.',
        steps: [
          'Load a regression dataset',
          'Implement gradient boosting with regression trees',
          'Tune learning rate and tree depth parameters',
          'Compare performance with random forest and linear regression',
          'Analyze feature importance'
        ],
        difficulty: 'advanced',
        xp: 500
      }
    ]
  },
  
  // XGBoost AlgoData
  {
    id: 'xgboost',
    title: 'XGBoost',
    category: 'ensemble',
    difficulty: 'advanced',
    tags: ['Gradient Boosting', 'Classification', 'Regression', 'Supervised'],
    description: 'An optimized gradient boosting implementation that is highly efficient, flexible, and portable, known for winning many machine learning competitions.',
    icon: 'arrow-up',
    lastUpdated: '2023-09-20',
    popularity: 0.96,
    
    concept: {
      overview: 'XGBoost (Extreme Gradient Boosting) is an advanced implementation of gradient boosted decision trees that uses second-order derivatives and regularization to optimize performance.',
      analogy: 'Like a team where each new member focuses on the mistakes of previous members, with a systematic approach to minimize errors.',
      history: 'Developed by Tianqi Chen in 2016 as part of the Distributed Machine Learning Community, quickly becoming a dominant algorithm in competitions.',
      mathematicalFormulation: {
        objectiveFunction: 'Obj(θ) = L(θ) + Ω(θ)',
        components: [
          {
            symbol: 'L(θ)',
            description: 'Training loss function (e.g., squared error, log loss)'
          },
          {
            symbol: 'Ω(θ)',
            description: 'Regularization term to prevent overfitting'
          }
        ],
        optimization: 'Uses second-order Taylor approximation and greedy algorithm for tree building',
        keyFeatures: [
          'Gradient boosting framework',
          'Regularized learning objective',
          'Tree pruning with max_depth parameter',
          'Handles missing values automatically',
          'Cross-validation at each iteration'
        ]
      },
      assumptions: [
        'Features should be predictive of target',
        'Data should be representative of population',
        'No strict distributional assumptions'
      ]
    },
    
    visualization: {
      visualizerKey: 'xgboost',
      defaultType: 'boosting-process',
      description: 'Interactive visualization of XGBoost algorithm. Observe how sequential trees focus on previous errors.',
      instructions: [
        'Adjust learning rate to control contribution of each tree',
        'Modify number of boosting rounds',
        'Observe how errors are reduced sequentially',
        'Toggle between different loss functions'
      ],
      showTypeSelector: true,
      types: [
        { 
          value: 'boosting-process', 
          label: 'Boosting Process', 
          description: 'Step-by-step visualization of sequential tree building',
          default: true 
        },
        { 
          value: 'loss-reduction', 
          label: 'Loss Reduction',
          description: 'Shows how error decreases with each iteration'
        },
        { 
          value: 'feature-importance', 
          label: 'Feature Importance',
          description: 'Visualize which features contribute most'
        },
        { 
          value: 'tree-structure', 
          label: 'Tree Structure',
          description: 'Examine individual tree components'
        },
        { 
          value: 'all', 
          label: 'Complete View',
          description: 'All visualization elements together'
        }
      ],
      parameters: {
        n_estimators: 50,
        learning_rate: 0.1,
        max_depth: 3,
        n_samples: 200,
        n_classes: 2,
        show_sequential: true,
        show_loss: true,
        show_importance: false,
        show_trees: false,
        animation_duration: 3000,
        interactive: true
      },
      performanceTips: [
        'Lower learning rate with more trees often gives better results',
        'Early stopping can prevent overfitting',
        'Feature importance helps with feature selection'
      ]
    },
    
    implementations: {
      python: {
        version: "3.9",
        libraries: ['xgboost@1.6.2', 'numpy@1.21.5', 'scikit-learn@1.0.2'],
        code: `# Python implementation using XGBoost
  import xgboost as xgb
  from sklearn.model_selection import train_test_split
  from sklearn.datasets import load_breast_cancer
  from sklearn.metrics import accuracy_score

  # Load sample data
  data = load_breast_cancer()
  X, y = data.data, data.target

  # Split data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Create DMatrix for efficiency
  dtrain = xgb.DMatrix(X_train, label=y_train)
  dtest = xgb.DMatrix(X_test, label=y_test)

  # Set parameters
  params = {
      'max_depth': 3,
      'eta': 0.1,
      'objective': 'binary:logistic',
      'eval_metric': 'logloss'
  }

  # Train model
  model = xgb.train(params, dtrain, num_boost_round=100,
                    evals=[(dtest, 'test')], early_stopping_rounds=10)

  # Predict
  y_pred = model.predict(dtest)
  y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]
  print(f"Accuracy: {accuracy_score(y_test, y_pred_binary):.3f}")`,
        timeComplexity: "O(n_estimators * n * p) where n is samples and p is features",
        spaceComplexity: "O(n)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "Use early_stopping_rounds to prevent overfitting. For multiclass problems, change objective to 'multi:softmax' or 'multi:softprob'."
      },
      r: {
        version: "4.2",
        libraries: ['xgboost@1.6.0.1'],
        code: `# R implementation using XGBoost
  library(xgboost)

  # Load sample data
  data(iris)
  # Binary classification for simplicity
  iris_binary <- iris[iris$Species != "setosa", ]
  iris_binary$Species <- as.factor(as.numeric(iris_binary$Species) - 2)

  # Prepare data
  features <- as.matrix(iris_binary[, 1:4])
  labels <- as.numeric(iris_binary$Species) - 1

  # Create DMatrix
  dtrain <- xgb.DMatrix(data = features, label = labels)

  # Set parameters
  params <- list(
    max_depth = 3,
    eta = 0.1,
    objective = "binary:logistic",
    eval_metric = "logloss"
  )

  # Train model
  model <- xgb.train(params, dtrain, nrounds = 100)

  # Predict
  predictions <- predict(model, features)
  predicted_labels <- ifelse(predictions > 0.5, 1, 0)
  print(table(predicted_labels, labels))`,
        timeComplexity: "O(n_estimators * n * p)",
        spaceComplexity: "O(n)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "The xgb.DMatrix format is more efficient for large datasets. Use cross-validation with xgb.cv for parameter tuning."
      },
      javascript: {
        version: "1.7.0",
        libraries: ['xgboost@0.4.5'],
        code: `// JavaScript implementation using XGBoost.js
  const xgboost = require('xgboost');

  // Sample training data
  const features = [
    [0, 0], [0, 1], [1, 0], [1, 1],
    [2, 2], [2, 3], [3, 2], [3, 3]
  ];
  const labels = [0, 0, 0, 0, 1, 1, 1, 1];

  // Create DMatrix
  const dtrain = new xgboost.DMatrix(features, labels);

  // Set parameters
  const params = {
    max_depth: 3,
    eta: 0.1,
    objective: 'binary:logistic',
    eval_metric: 'logloss'
  };

  // Train model
  const booster = await xgboost.train(params, dtrain, 100);

  // Predict
  const test = [[0.5, 0.5], [2.5, 2.5]];
  const dtest = new xgboost.DMatrix(test);
  const predictions = await booster.predict(dtest);
  console.log('Predictions:', predictions);`,
        timeComplexity: "O(n_estimators * n * p)",
        spaceComplexity: "O(n)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "XGBoost.js is a WebAssembly port of XGBoost. For Node.js, consider using the native binding version for better performance."
      },
      cpp: {
        version: "17",
        libraries: ['xgboost@1.6.2'],
        code: `// C++ implementation using XGBoost
  #include <xgboost/c_api.h>
  #include <vector>
  #include <iostream>

  int main() {
    // Sample data
    std::vector<float> data = {0,0, 0,1, 1,0, 1,1, 2,2, 2,3, 3,2, 3,3};
    std::vector<float> labels = {0, 0, 0, 0, 1, 1, 1, 1};
    
    // Create DMatrix
    DMatrixHandle dtrain;
    XGDMatrixCreateFromMat(data.data(), 8, 2, -1, &dtrain);
    XGDMatrixSetFloatInfo(dtrain, "label", labels.data(), 8);
    
    // Set parameters
    const char* params[] = {
      "max_depth", "3",
      "eta", "0.1",
      "objective", "binary:logistic",
      "eval_metric", "logloss",
      NULL
    };
    
    // Create booster
    BoosterHandle booster;
    XGBoosterCreate(&dtrain, 1, &booster);
    XGBoosterSetParam(booster, "max_depth", "3");
    XGBoosterSetParam(booster, "eta", "0.1");
    XGBoosterSetParam(booster, "objective", "binary:logistic");
    
    // Train for 100 rounds
    for (int i = 0; i < 100; i++) {
      XGBoosterUpdateOneIter(booster, i, dtrain);
    }
    
    // Cleanup
    XGDMatrixFree(dtrain);
    XGBoosterFree(booster);
    
    return 0;
  }`,
        timeComplexity: "O(n_estimators * n * p)",
        spaceComplexity: "O(n)",
        author: {
          name: "Numerical Computing Team"
        },
        lastUpdated: "2023-03-05",
        notes: "The C API provides low-level control. For most applications, the Python or R interfaces are recommended for easier use."
      }
    },

    prosCons: {
      strengths: [
        'State-of-the-art performance on many problems',
        'Handles missing values automatically',
        'Provides feature importance',
        'Extensive regularization to prevent overfitting'
      ],
      weaknesses: [
        'Can be more complex to tune than other algorithms',
        'Sequential training is harder to parallelize',
        'More prone to overfitting without proper regularization',
        'Less interpretable than simpler models'
      ]
    },
    
    hyperparameters: [
      {
        name: 'learning_rate',
        type: 'range',
        min: 0.01,
        max: 0.3,
        step: 0.01,
        default: 0.1,
        description: 'Step size shrinkage used in update to prevent overfitting'
      },
      {
        name: 'max_depth',
        type: 'range',
        min: 1,
        max: 10,
        step: 1,
        default: 3,
        description: 'Maximum depth of a tree'
      },
      {
        name: 'subsample',
        type: 'range',
        min: 0.5,
        max: 1.0,
        step: 0.1,
        default: 1.0,
        description: 'Subsample ratio of the training instances'
      },
      {
        name: 'colsample_bytree',
        type: 'range',
        min: 0.5,
        max: 1.0,
        step: 0.1,
        default: 1.0,
        description: 'Subsample ratio of columns when constructing each tree'
      }
    ],
    
    useCases: [
      {
        title: 'Competition Winning Solutions',
        description: 'Many winning solutions in Kaggle and other data science competitions use XGBoost.',
        dataset: 'Various Competition Datasets'
      },
      {
        title: 'Click-Through Rate Prediction',
        description: 'Predicting whether users will click on online advertisements.',
        dataset: 'Criteo Display Advertising'
      }
    ],
    
    comparisons: [
      {
        algorithm: 'Random Forest',
        comparison: 'XGBoost builds trees sequentially focusing on errors, while Random Forest builds trees independently'
      },
      {
        algorithm: 'LightGBM',
        comparison: 'XGBoost is more established but LightGBM can be faster and use less memory on large datasets'
      }
    ],
    
    quiz: [
      {
        question: 'What makes XGBoost different from traditional gradient boosting?',
        options: [
          'Regularization and second-order derivatives',
          'Use of neural networks as weak learners',
          'Bayesian optimization of parameters',
          'Exclusive focus on linear models'
        ],
        correct: 0,
        explanation: 'XGBoost adds regularization to the objective function and uses second-order Taylor approximation for more accurate optimization.'
      },
      {
        question: 'What is the purpose of the learning_rate parameter in XGBoost?',
        options: [
          'To control the contribution of each tree to the final prediction',
          'To determine the splitting criterion at each node',
          'To set the threshold for early stopping',
          'To control the randomness in feature selection'
        ],
        correct: 0,
        explanation: 'The learning rate (eta) shrinks the contribution of each tree to prevent overfitting and allow more conservative boosting.'
      }
    ],
    
    projects: [
      {
        title: 'House Price Prediction',
        description: 'Build an XGBoost model to predict house prices for the Kaggle competition.',
        steps: [
          'Load and explore the Ames Housing dataset',
          'Perform extensive feature engineering',
          'Tune XGBoost hyperparameters using cross-validation',
          'Use early stopping to prevent overfitting',
          'Submit predictions to Kaggle leaderboard'
        ],
        difficulty: 'advanced',
        xp: 500
      }
    ]
  },

  // Neural Network AlgoData
  {
    id: 'neural-network',
    title: 'Neural Network',
    category: 'deep',
    difficulty: 'beginner',
    tags: ['Deep Learning', 'Classification', 'Regression', 'Supervised'],
    description: 'A computational model inspired by biological neural networks, consisting of interconnected nodes that process information through multiple layers.',
    icon: 'brain',
    lastUpdated: '2023-08-15',
    popularity: 0.92,
    
    concept: {
      overview: 'Neural networks are composed of layers of artificial neurons that learn hierarchical representations of data through forward propagation and backpropagation.',
      analogy: 'Like a team of specialists where each layer extracts increasingly complex features - from simple edges to complex patterns in images.',
      history: 'First conceptualized in the 1940s, with major breakthroughs including the perceptron (1958), backpropagation (1986), and deep learning revolution (2010s).',
      mathematicalFormulation: {
        neuron: 'z = σ(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)',
        activationFunctions: [
          { name: 'Sigmoid', formula: 'σ(x) = 1 / (1 + e⁻ˣ)', description: 'Squashes values between 0 and 1' },
          { name: 'ReLU', formula: 'f(x) = max(0, x)', description: 'Most common in deep learning' },
          { name: 'Tanh', formula: 'tanh(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)', description: 'Squashes values between -1 and 1' }
        ],
        costFunction: 'Cross-Entropy = -Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]',
        optimization: 'Typically optimized using variants of Gradient Descent (Adam, SGD, RMSprop)'
      },
      assumptions: [
        'No specific distributional assumptions',
        'Large amounts of data typically required',
        'Features should be normalized for best performance'
      ]
    },
    
    visualization: {
      visualizerKey: 'neural-network',
      defaultType: 'architecture',
      description: 'Interactive visualization of neural network architecture and training process.',
      instructions: [
        'Add/remove layers to customize network architecture',
        'Adjust learning rate to see training convergence',
        'Toggle activation functions to see their effects',
        'Watch how weights update during backpropagation'
      ],
      showTypeSelector: true,
      types: [
        { 
          value: 'architecture', 
          label: 'Network Architecture', 
          description: 'Visualize the structure of neural network layers',
          default: true 
        },
        { 
          value: 'training-process', 
          label: 'Training Process',
          description: 'Watch weights update during backpropagation'
        },
        { 
          value: 'decision-boundary', 
          label: 'Decision Boundary',
          description: 'See how complex boundaries are learned'
        },
        { 
          value: 'feature-visualization', 
          label: 'Feature Learning',
          description: 'Visualize what each layer learns'
        },
        { 
          value: 'all', 
          label: 'Complete View',
          description: 'All visualization elements together'
        }
      ],
      parameters: {
        n_layers: 3,
        layer_sizes: [4, 8, 2],
        activation: 'relu',
        learning_rate: 0.01,
        epochs: 100,
        batch_size: 32,
        show_weights: true,
        show_activations: false,
        show_gradients: false,
        animation_duration: 3000,
        interactive: true
      },
      performanceTips: [
        'Start with simple architectures for faster visualization',
        'Higher learning rates may cause unstable training',
        'Visualizing gradients is computationally intensive'
      ]
    },
    
    implementations: {
      python: {
        version: "3.9",
        libraries: ['tensorflow@2.10.0', 'keras@2.10.0', 'numpy@1.21.5'],
        code: `# Python implementation using Keras
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense
  from sklearn.model_selection import train_test_split
  from sklearn.datasets import make_classification
  import numpy as np

  # Generate sample data
  X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

  # Split data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Create neural network
  model = Sequential([
      Dense(64, activation='relu', input_shape=(20,)),
      Dense(32, activation='relu'),
      Dense(1, activation='sigmoid')
  ])

  # Compile model
  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

  # Train model
  history = model.fit(X_train, y_train, 
                    epochs=50, 
                    batch_size=32,
                    validation_split=0.2)

  # Evaluate
  test_loss, test_acc = model.evaluate(X_test, y_test)
  print(f"Test accuracy: {test_acc:.3f}")`,
        timeComplexity: "O(n × l × nₗ) where l is layers and nₗ is neurons per layer",
        spaceComplexity: "O(n × l × nₗ)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For regression tasks, use linear activation in output layer and MSE loss. For multiclass, use softmax and categorical crossentropy."
      },
      r: {
        version: "4.2",
        libraries: ['keras@2.9.0', 'tensorflow@2.9.0'],
        code: `# R implementation using Keras
  library(keras)
  library(tensorflow)

  # Generate sample data
  data <- dataset_boston_housing()
  X_train <- data$train$x
  y_train <- data$train$y
  X_test <- data$test$x
  y_test <- data$test$y

  # Create neural network
  model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = 'relu', input_shape = dim(X_train)[2]) %>%
    layer_dense(units = 32, activation = 'relu') %>%
    layer_dense(units = 1)

  # Compile model
  model %>% compile(
    optimizer = 'adam',
    loss = 'mse',
    metrics = c('mae')
  )

  # Train model
  history <- model %>% fit(
    X_train, y_train,
    epochs = 50,
    batch_size = 32,
    validation_split = 0.2,
    verbose = 0
  )

  # Evaluate
  model %>% evaluate(X_test, y_test)`,
        timeComplexity: "O(n × l × nₗ)",
        spaceComplexity: "O(n × l × nₗ)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "Make sure to install Python and TensorFlow before using Keras in R. For classification, change output activation and loss function."
      },
      javascript: {
        version: "1.7.0",
        libraries: ['tensorflow.js@3.18.0'],
        code: `// JavaScript implementation using TensorFlow.js
  import * as tf from '@tensorflow/tfjs';

  // Generate sample data
  const xs = tf.randomNormal([100, 20]);
  const ys = tf.randomUniform([100, 1], 0, 2, 'int32');

  // Define neural network
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 64, activation: 'relu', inputShape: [20]}));
  model.add(tf.layers.dense({units: 32, activation: 'relu'}));
  model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));

  // Prepare for training
  model.compile({
    optimizer: 'adam',
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });

  // Train model
  await model.fit(xs, ys, {
    epochs: 50,
    batchSize: 32,
    validationSplit: 0.2
  });

  // Predict
  const sample = tf.randomNormal([1, 20]);
  model.predict(sample).print();`,
        timeComplexity: "O(n × l × nₗ)",
        spaceComplexity: "O(n × l × nₗ)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For Node.js, use '@tensorflow/tfjs-node' for better performance. WebGL acceleration is available in browsers."
      },
      cpp: {
        version: "17",
        libraries: ['mlpack@3.4.2', 'armadillo@11.2.1'],
        code: `// C++ implementation using mlpack
  #include <mlpack.hpp>
  #include <ensmallen.hpp>

  using namespace mlpack;
  using namespace arma;

  int main() {
    // Sample data (features and labels)
    mat features = randu<mat>(20, 1000); // 20 features, 1000 samples
    Row<size_t> labels = randi<Row<size_t>>(1000, distr_param(0, 1)); // Binary labels
    
    // Initialize neural network
    FFN<NegativeLogLikelihood<>, RandomInitialization> model;
    
    // Add layers
    model.Add<Linear<>>(20, 64); // Input to hidden
    model.Add<ReLULayer<>>();
    model.Add<Linear<>>(64, 32); // Hidden to hidden
    model.Add<ReLULayer<>>();
    model.Add<Linear<>>(32, 2);  // Hidden to output
    model.Add<LogSoftMax<>>();
    
    // Train model
    ens::Adam optimizer(0.01, 32, 0.9, 0.999, 1e-8, 50, 1e-8, true);
    model.Train(features, labels, optimizer);
    
    // Predict
    mat testPoint = randu<mat>(20, 1);
    mat predictions;
    model.Predict(testPoint, predictions);
    
    std::cout << "Predictions: " << predictions.t() << std::endl;
    return 0;
  }`,
        timeComplexity: "O(n × l × nₗ)",
        spaceComplexity: "O(n × l × nₗ)",
        author: {
          name: "Numerical Computing Team"
        },
        lastUpdated: "2023-03-05",
        notes: "mlpack provides flexible neural network implementation. For more complex architectures, consider using different layer types and activation functions."
      }
    },

    prosCons: {
      strengths: [
        'Can approximate any complex function (universal approximation theorem)',
        'Automatically learns feature representations',
        'Excellent performance with sufficient data',
        'Flexible for various problem types'
      ],
      weaknesses: [
        'Computationally expensive to train',
        'Requires large amounts of data',
        'Black box nature (hard to interpret)',
        'Sensitive to hyperparameter choices'
      ]
    },
    
    hyperparameters: [
      {
        name: 'hidden_layers',
        type: 'range',
        min: 1,
        max: 10,
        step: 1,
        default: 3,
        description: 'Number of hidden layers in the network'
      },
      {
        name: 'learning_rate',
        type: 'range',
        min: 0.0001,
        max: 0.1,
        step: 0.0001,
        default: 0.001,
        description: 'Step size for weight updates during optimization'
      },
      {
        name: 'activation',
        type: 'select',
        options: ['relu', 'sigmoid', 'tanh', 'leaky_relu'],
        default: 'relu',
        description: 'Activation function for hidden layers'
      }
    ],
    
    useCases: [
      {
        title: 'Image Classification',
        description: 'Classifying images into categories using convolutional neural networks.',
        dataset: 'CIFAR-10, ImageNet'
      },
      {
        title: 'Natural Language Processing',
        description: 'Text classification, sentiment analysis, and machine translation.',
        dataset: 'IMDB Reviews, Wikipedia Text'
      }
    ],
    
    comparisons: [
      {
        algorithm: 'Linear Models',
        comparison: 'Neural networks can model complex nonlinear relationships while linear models are limited to linear relationships'
      },
      {
        algorithm: 'Decision Trees',
        comparison: 'Neural networks typically require more data but can achieve higher performance on complex problems'
      }
    ],
    
    quiz: [
      {
        question: 'What is the purpose of the activation function in a neural network?',
        options: [
          'To increase computational speed',
          'To introduce non-linearity into the model',
          'To reduce memory usage',
          'To normalize the input data'
        ],
        correct: 1,
        explanation: 'Activation functions introduce non-linearity, allowing neural networks to learn complex patterns.'
      },
      {
        question: 'Which algorithm is commonly used to train neural networks?',
        options: [
          'K-means clustering',
          'Backpropagation',
          'Principal Component Analysis',
          'Support Vector Machines'
        ],
        correct: 1,
        explanation: 'Backpropagation is the standard algorithm for training neural networks by propagating errors backward.'
      }
    ],
    
    projects: [
      {
        title: 'MNIST Digit Classifier',
        description: 'Build a neural network to classify handwritten digits from the MNIST dataset.',
        steps: [
          'Load and preprocess MNIST dataset',
          'Design neural network architecture',
          'Train model with appropriate hyperparameters',
          'Evaluate performance on test set',
          'Visualize learned features and misclassifications'
        ],
        difficulty: 'intermediate',
        xp: 400
      }
    ]
  },

  // Convolutional Neural Network AlgoData
  {
    id: 'cnn',
    title: 'Convolutional Neural Network',
    category: 'deep',
    difficulty: 'intermediate',
    tags: ['Deep Learning', 'Computer Vision', 'Image Processing', 'Supervised'],
    description: 'A specialized neural network architecture designed for processing grid-like data such as images, using convolutional layers to extract spatial features.',
    icon: 'image',
    lastUpdated: '2023-08-15',
    popularity: 0.92,
    
    concept: {
      overview: 'CNNs are designed to automatically and adaptively learn spatial hierarchies of features through backpropagation by using a building block called convolutional layers.',
      analogy: 'Like how the visual cortex processes visual information through receptive fields, CNNs use filters to detect patterns at different scales and complexities.',
      history: 'Originated from the neocognitron (1980) and popularized by LeNet-5 (1998) for digit recognition. Revolutionized computer vision after AlexNet (2012) won ImageNet.',
      mathematicalFormulation: {
        convolutionOperation: 'S(i,j) = (I*K)(i,j) = ΣₘΣₙI(i+m,j+n)K(m,n)',
        variables: [
          { symbol: 'I', description: 'Input matrix (image)' },
          { symbol: 'K', description: 'Kernel/filter matrix' },
          { symbol: 'S', description: 'Feature map output' }
        ],
        poolingOperation: 'Max Pooling: P(i,j) = max(S(2i:2i+1, 2j:2j+1))',
        activationFunction: 'Typically ReLU: f(x) = max(0,x)',
        architectureComponents: [
          'Convolutional Layers (feature extraction)',
          'Pooling Layers (dimensionality reduction)',
          'Fully Connected Layers (classification)',
          'Activation Functions (non-linearity)'
        ]
      },
      assumptions: [
        'Local connectivity (spatial locality)',
        'Parameter sharing (translation invariance)',
        'Hierarchical feature learning',
        'Stationarity of statistics'
      ]
    },
    
    visualization: {
      visualizerKey: 'cnn',
      defaultType: 'feature-maps',
      description: 'Interactive visualization of CNN architecture and feature extraction process. Observe how different layers detect increasingly complex patterns.',
      instructions: [
        'Adjust network depth to see hierarchical feature learning',
        'Modify filter sizes and strides to understand convolution operations',
        'Toggle between different input images to see generalization',
        'Enable feature map visualization to see what each layer detects'
      ],
      showTypeSelector: true,
      types: [
        { 
          value: 'feature-maps', 
          label: 'Feature Maps', 
          description: 'Visualize activations at different layers',
          default: true 
        },
        { 
          value: 'architecture', 
          label: 'Network Architecture',
          description: 'Interactive diagram of CNN layers'
        },
        { 
          value: 'filter-visualization', 
          label: 'Filter Visualization',
          description: 'See what patterns each filter detects'
        },
        { 
          value: 'training-process', 
          label: 'Training Process',
          description: 'Watch the network learn features over time'
        },
        { 
          value: 'all', 
          label: 'Complete View',
          description: 'All visualization elements together'
        }
      ],
      parameters: {
        network_depth: 4,
        filter_size: 3,
        num_filters: 32,
        stride: 1,
        padding: 'same',
        pooling_type: 'max',
        activation: 'relu',
        input_image: 'mnist-digit',
        show_feature_maps: true,
        show_architecture: false,
        show_filters: false,
        show_training: false,
        animation_duration: 2500,
        interactive: true
      },
      performanceTips: [
        'Deeper networks require more computation - reduce depth for faster visualization',
        'Feature map visualization is memory intensive with many filters',
        'Training visualization shows the most interesting process but is slowest'
      ]
    },
    
    implementations: {
      python: {
        version: "3.9",
        libraries: ['tensorflow@2.9.0', 'keras@2.9.0', 'numpy@1.21.5'],
        code: `# Python implementation using TensorFlow/Keras
  import tensorflow as tf
  from tensorflow.keras import datasets, layers, models

  # Load CIFAR-10 dataset
  (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

  # Normalize pixel values
  train_images, test_images = train_images / 255.0, test_images / 255.0

  # Create CNN model
  model = models.Sequential()
  model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.Flatten())
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(10))

  # Compile and train
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  history = model.fit(train_images, train_labels, epochs=10, 
                      validation_data=(test_images, test_labels))

  # Evaluate
  test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
  print(f"Test accuracy: {test_acc}")`,
        timeComplexity: "O(n × k × f² × c_in × c_out) per layer",
        spaceComplexity: "O(n × h × w × c)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For larger images, consider using transfer learning with pre-trained models like ResNet or EfficientNet. Use data augmentation to improve generalization."
      },
      r: {
        version: "4.2",
        libraries: ['keras@2.9.0', 'tensorflow@2.9.0'],
        code: `# R implementation using Keras
  library(keras)

  # Load CIFAR-10 dataset
  cifar10 <- dataset_cifar10()
  x_train <- cifar10$train$x / 255
  y_train <- cifar10$train$y
  x_test <- cifar10$test$x / 255
  y_test <- cifar10$test$y

  # Create CNN model
  model <- keras_model_sequential() %>%
    layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', 
                  input_shape = c(32, 32, 3)) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = 'relu') %>%
    layer_dense(units = 10, activation = 'softmax')

  # Compile and train
  model %>% compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = c('accuracy')
  )

  history <- model %>% fit(
    x_train, y_train,
    epochs = 10,
    validation_data = list(x_test, y_test)
  )

  # Evaluate
  score <- model %>% evaluate(x_test, y_test)
  cat('Test accuracy:', score$acc, '\n')`,
        timeComplexity: "O(n × k × f² × c_in × c_out) per layer",
        spaceComplexity: "O(n × h × w × c)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "R interface to Keras provides the same functionality as Python. For GPU acceleration, ensure proper TensorFlow installation with CUDA support."
      },
      javascript: {
        version: "1.7.0",
        libraries: ['tensorflow.js@3.18.0'],
        code: `// JavaScript implementation using TensorFlow.js
  import * as tf from '@tensorflow/tfjs';

  // Create a simple CNN for MNIST
  function createModel() {
    const model = tf.sequential();
    
    // First convolutional layer
    model.add(tf.layers.conv2d({
      inputShape: [28, 28, 1],
      filters: 32,
      kernelSize: 3,
      activation: 'relu'
    }));
    
    // First max pooling layer
    model.add(tf.layers.maxPooling2d({poolSize: 2}));
    
    // Second convolutional layer
    model.add(tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: 'relu'
    }));
    
    // Second max pooling layer
    model.add(tf.layers.maxPooling2d({poolSize: 2}));
    
    // Flatten and dense layers
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({units: 64, activation: 'relu'}));
    model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
    
    return model;
  }

  // Compile model
  const model = createModel();
  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  // Note: Training code would require loading and preprocessing MNIST data
  console.log('CNN model created successfully');`,
        timeComplexity: "O(n × k × f² × c_in × c_out) per layer",
        spaceComplexity: "O(n × h × w × c)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For browser-based applications, consider using smaller models or quantized weights. WebGL acceleration is automatically used when available."
      },
      cpp: {
        version: "17",
        libraries: ['OpenCV@4.5.5', 'Eigen@3.4.0'],
        code: `// C++ implementation using OpenCV's DNN module
  #include <opencv2/opencv.hpp>
  #include <opencv2/dnn.hpp>
  #include <iostream>

  using namespace cv;
  using namespace dnn;
  using namespace std;

  int main() {
    // Load pre-trained model (example with OpenCV's DNN module)
    String modelConfig = "model.prototxt";
    String modelWeights = "model.caffemodel";
    
    Net net = readNetFromCaffe(modelConfig, modelWeights);
    
    // Load image
    Mat image = imread("image.jpg");
    Mat inputBlob = blobFromImage(image, 1.0, Size(224, 224), Scalar(104, 117, 123));
    
    // Set input and forward pass
    net.setInput(inputBlob);
    Mat detection = net.forward();
    
    // Process results
    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(detection, &minVal, &maxVal, &minLoc, &maxLoc);
    
    cout << "Predicted class: " << maxLoc.x << " with confidence: " << maxVal << endl;
    
    return 0;
  }`,
        timeComplexity: "O(n × k × f² × c_in × c_out) per layer",
        spaceComplexity: "O(n × h × w × c)",
        author: {
          name: "Numerical Computing Team"
        },
        lastUpdated: "2023-03-05",
        notes: "For custom CNN architectures in C++, consider using libraries like tiny-dnn or implementing layers manually using Eigen for tensor operations."
      }
    },

    prosCons: {
      strengths: [
        'Automatic feature extraction from raw data',
        'Spatial invariance to translation, rotation, and scaling',
        'Parameter sharing reduces number of parameters',
        'Excellent performance on image data'
      ],
      weaknesses: [
        'Computationally intensive during training',
        'Requires large amounts of labeled data',
        'Black box nature makes interpretation difficult',
        'Sensitive to adversarial attacks'
      ]
    },
    
    hyperparameters: [
      {
        name: 'filters',
        type: 'range',
        min: 16,
        max: 512,
        step: 16,
        default: 32,
        description: 'Number of filters in convolutional layer'
      },
      {
        name: 'kernel_size',
        type: 'range',
        min: 1,
        max: 11,
        step: 2,
        default: 3,
        description: 'Size of convolutional kernel (typically 3x3 or 5x5)'
      },
      {
        name: 'pool_size',
        type: 'range',
        min: 2,
        max: 4,
        step: 1,
        default: 2,
        description: 'Size of pooling window (typically 2x2)'
      }
    ],
    
    useCases: [
      {
        title: 'Image Classification',
        description: 'Classifying images into categories such as animals, objects, or scenes.',
        dataset: 'ImageNet, CIFAR-10/100'
      },
      {
        title: 'Object Detection',
        description: 'Detecting and localizing multiple objects within an image.',
        dataset: 'COCO, Pascal VOC'
      },
      {
        title: 'Medical Image Analysis',
        description: 'Analyzing medical images for disease detection and diagnosis.',
        dataset: 'CheXpert, MIMIC-CXR'
      }
    ],
    
    comparisons: [
      {
        algorithm: 'Fully Connected Networks',
        comparison: 'CNNs use parameter sharing and local connectivity, making them more efficient and effective for spatial data than fully connected networks'
      },
      {
        algorithm: 'Vision Transformers',
        comparison: 'CNNs use inductive biases for spatial processing while Vision Transformers use self-attention mechanisms, with ViTs often requiring more data but achieving state-of-the-art results'
      }
    ],
    
    quiz: [
      {
        question: 'What is the primary purpose of convolutional layers in CNNs?',
        options: [
          'To reduce dimensionality of the input',
          'To extract spatial features from input data',
          'To classify the final output',
          'To normalize the input data'
        ],
        correct: 1,
        explanation: 'Convolutional layers apply filters to input data to detect spatial patterns and features at different scales.'
      },
      {
        question: 'Which operation helps CNNs achieve translation invariance?',
        options: [
          'Convolution',
          'Pooling',
          'Activation functions',
          'Fully connected layers'
        ],
        correct: 1,
        explanation: 'Pooling operations reduce the spatial dimensions of feature maps, making the network less sensitive to the exact position of features.'
      }
    ],
    
    projects: [
      {
        title: 'Handwritten Digit Recognition',
        description: 'Build a CNN to classify handwritten digits from the MNIST dataset.',
        steps: [
          'Load and preprocess MNIST dataset',
          'Design CNN architecture with convolutional and pooling layers',
          'Train the model with appropriate hyperparameters',
          'Evaluate performance on test set',
          'Visualize feature maps to understand what the network learns'
        ],
        difficulty: 'intermediate',
        xp: 400
      }
    ]
  },

  // Recurrent Neural Network AlgoData
  {
    id: 'rnn',
    title: 'Recurrent Neural Network',
    category: 'deep',
    difficulty: 'intermediate',
    tags: ['Deep Learning', 'Time Series', 'Sequences', 'Supervised'],
    description: 'A class of neural networks designed for sequential data where connections between nodes form a directed graph along a temporal sequence.',
    icon: 'history',
    lastUpdated: '2023-09-20',
    popularity: 0.85,
    
    concept: {
      overview: 'RNNs maintain a hidden state that captures information about previous elements in a sequence, allowing them to exhibit temporal dynamic behavior.',
      analogy: 'Like reading a book where you understand each word based on your memory of previous words, RNNs process sequences while maintaining context.',
      history: 'First proposed in the 1980s, with significant developments including LSTM (1997) and GRU (2014) to address vanishing gradient problems.',
      mathematicalFormulation: {
        basicRNN: 'h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)',
        output: 'y_t = W_hy * h_t + b_y',
        variables: [
          { symbol: 'h_t', description: 'Hidden state at time t' },
          { symbol: 'x_t', description: 'Input at time t' },
          { symbol: 'y_t', description: 'Output at time t' },
          { symbol: 'W', description: 'Weight matrices' },
          { symbol: 'b', description: 'Bias vectors' }
        ],
        advancedVariants: [
          'LSTM: Long Short-Term Memory with forget, input, and output gates',
          'GRU: Gated Recurrent Unit with reset and update gates',
          'Bidirectional RNN: Processes sequence in both directions'
        ]
      },
      assumptions: [
        'Sequential dependency in data',
        'Stationarity of temporal patterns',
        'Fixed or variable length sequences',
        'Temporal ordering matters'
      ]
    },
    
    visualization: {
      visualizerKey: 'rnn',
      defaultType: 'unrolled',
      description: 'Interactive visualization of RNN architecture and sequence processing. See how hidden states evolve over time and capture temporal dependencies.',
      instructions: [
        'Adjust sequence length to see unrolling over time',
        'Modify hidden state size to change model capacity',
        'Toggle between different RNN variants (Simple RNN, LSTM, GRU)',
        'Enable gradient flow visualization to understand vanishing/exploding gradients'
      ],
      showTypeSelector: true,
      types: [
        { 
          value: 'unrolled', 
          label: 'Unrolled View', 
          description: 'Shows the network unrolled across time steps',
          default: true 
        },
        { 
          value: 'rolled', 
          label: 'Rolled View',
          description: 'Compact representation of recurrent connections'
        },
        { 
          value: 'hidden-states', 
          label: 'Hidden State Evolution',
          description: 'Visualizes how hidden states change over time'
        },
        { 
          value: 'gate-operations', 
          label: 'Gate Operations (LSTM/GRU)',
          description: 'Shows internal gating mechanisms'
        },
        { 
          value: 'all', 
          label: 'Complete View',
          description: 'All visualization elements together'
        }
      ],
      parameters: {
        sequence_length: 10,
        hidden_size: 32,
        rnn_type: 'lstm',
        input_size: 8,
        output_size: 1,
        show_unrolled: true,
        show_rolled: false,
        show_hidden_states: false,
        show_gates: false,
        animation_duration: 3000,
        interactive: true
      },
      performanceTips: [
        'Longer sequences require more computation and memory',
        'LSTM and GRU visualizations show more internal details',
        'Hidden state visualization helps understand memory retention'
      ]
    },
    
    implementations: {
      python: {
        version: "3.9",
        libraries: ['tensorflow@2.9.0', 'keras@2.9.0', 'numpy@1.21.5'],
        code: `# Python implementation using TensorFlow/Keras
  import tensorflow as tf
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import LSTM, Dense, Embedding
  import numpy as np

  # Generate sample sequence data
  def generate_time_series(n_steps, n_samples=1000):
      freq1, freq2, offsets1, offsets2 = np.random.rand(4, n_samples, 1)
      time = np.linspace(0, 1, n_steps)
      series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  # Wave 1
      series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # Wave 2
      series += 0.1 * (np.random.rand(n_samples, n_steps) - 0.5)    # Noise
      return series[..., np.newaxis].astype(np.float32)

  # Prepare data
  n_steps = 50
  n_samples = 10000
  series = generate_time_series(n_steps, n_samples)
  X_train, y_train = series[:7000, :n_steps-1], series[:7000, -1]
  X_val, y_val = series[7000:9000, :n_steps-1], series[7000:9000, -1]
  X_test, y_test = series[9000:, :n_steps-1], series[9000:, -1]

  # Create LSTM model
  model = Sequential([
      LSTM(50, return_sequences=True, input_shape=[None, 1]),
      LSTM(50),
      Dense(1)
  ])

  model.compile(optimizer='adam', loss='mse', metrics=['mae'])

  # Train model
  history = model.fit(X_train, y_train, epochs=20,
                      validation_data=(X_val, y_val))

  # Evaluate
  test_loss, test_mae = model.evaluate(X_test, y_test)
  print(f"Test MAE: {test_mae}")`,
        timeComplexity: "O(t × n²) where t is sequence length and n is hidden size",
        spaceComplexity: "O(t × n)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For very long sequences, consider using CuDNNLSTM for GPU acceleration or switching to transformer architectures. Gradient clipping helps with exploding gradients."
      },
      r: {
        version: "4.2",
        libraries: ['keras@2.9.0', 'tensorflow@2.9.0'],
        code: `# R implementation using Keras
  library(keras)
  library(tensorflow)

  # Generate sample sequence data
  generate_series <- function(n_steps, n_samples = 1000) {
    time <- seq(0, 1, length.out = n_steps)
    series <- array(dim = c(n_samples, n_steps, 1))
    
    for (i in 1:n_samples) {
      freq1 <- runif(1, 10, 20)
      freq2 <- runif(1, 20, 30)
      offset1 <- runif(1)
      offset2 <- runif(1)
      
      series[i,,1] <- 0.5 * sin((time - offset1) * freq1) + 
                      0.2 * sin((time - offset2) * freq2) + 
                      0.1 * (runif(n_steps) - 0.5)
    }
    
    return(series)
  }

  # Prepare data
  n_steps <- 50
  series <- generate_series(n_steps, 1000)
  X_train <- series[1:700, 1:(n_steps-1), , drop = FALSE]
  y_train <- series[1:700, n_steps, ]
  X_val <- series[701:900, 1:(n_steps-1), , drop = FALSE]
  y_val <- series[701:900, n_steps, ]
  X_test <- series[901:1000, 1:(n_steps-1), , drop = FALSE]
  y_test <- series[901:1000, n_steps, ]

  # Create LSTM model
  model <- keras_model_sequential() %>%
    layer_lstm(units = 50, return_sequences = TRUE, input_shape = c(NULL, 1)) %>%
    layer_lstm(units = 50) %>%
    layer_dense(units = 1)

  model %>% compile(
    optimizer = 'adam',
    loss = 'mse',
    metrics = 'mae'
  )

  # Train model
  history <- model %>% fit(
    X_train, y_train,
    epochs = 20,
    validation_data = list(X_val, y_val)
  )

  # Evaluate
  score <- model %>% evaluate(X_test, y_test)
  cat('Test MAE:', score$mae, '\n')`,
        timeComplexity: "O(t × n²) where t is sequence length and n is hidden size",
        spaceComplexity: "O(t × n)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For text data, use embedding layers before RNN layers. Bidirectional RNNs often perform better for many NLP tasks."
      },
      javascript: {
        version: "1.7.0",
        libraries: ['tensorflow.js@3.18.0'],
        code: `// JavaScript implementation using TensorFlow.js
  import * as tf from '@tensorflow/tfjs';

  // Create a simple RNN model
  function createModel(rnnType = 'simple', hiddenSize = 32) {
    const model = tf.sequential();
    
    // Add RNN layer based on type
    if (rnnType === 'simple') {
      model.add(tf.layers.simpleRNN({
        units: hiddenSize,
        returnSequences: false,
        inputShape: [null, 1] // [timesteps, features]
      }));
    } else if (rnnType === 'lstm') {
      model.add(tf.layers.lstm({
        units: hiddenSize,
        returnSequences: false,
        inputShape: [null, 1]
      }));
    } else if (rnnType === 'gru') {
      model.add(tf.layers.gru({
        units: hiddenSize,
        returnSequences: false,
        inputShape: [null, 1]
      }));
    }
    
    // Add output layer
    model.add(tf.layers.dense({units: 1}));
    
    return model;
  }

  // Compile model
  const model = createModel('lstm', 32);
  model.compile({
    optimizer: 'adam',
    loss: 'meanSquaredError',
    metrics: ['mae']
  });

  console.log('RNN model created successfully');

  // Note: Training would require sequential data preparation
  // For text data, use tokenization and embedding layers`,
        timeComplexity: "O(t × n²) where t is sequence length and n is hidden size",
        spaceComplexity: "O(t × n)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For browser deployment, consider model quantization or using smaller hidden sizes. TensorFlow.js supports GPU acceleration through WebGL."
      },
      cpp: {
        version: "17",
        libraries: ['TensorFlow@2.9.0', 'Eigen@3.4.0'],
        code: `// C++ implementation using TensorFlow C++ API
  #include <tensorflow/cc/client/client_session.h>
  #include <tensorflow/cc/ops/standard_ops.h>
  #include <tensorflow/core/framework/tensor.h>
  #include <iostream>

  using namespace tensorflow;
  using namespace tensorflow::ops;

  int main() {
    Scope root = Scope::NewRootScope();
    
    // Placeholders for input and targets
    auto input = Placeholder(root, DT_FLOAT, Placeholder::Shape({-1, -1, 1})); // [batch, time, features]
    auto target = Placeholder(root, DT_FLOAT, Placeholder::Shape({-1, 1}));
    
    // Create LSTM cell
    auto lstm_cell = ops::LSTMBlockCell(root, 32, 0.1, "lstm_cell");
    
    // RNN layer (simplified example - actual implementation would be more complex)
    auto rnn_output = ops::DynamicRnn(root, lstm_cell, input, DT_FLOAT, "rnn");
    
    // Get last output
    auto last_output = ops::Gather(root, rnn_output.output, -1, Gather::Axis(1));
    auto squeezed = ops::Squeeze(root, last_output, Squeeze::Axis({1}));
    
    // Output layer
    auto output = ops::Dense(root, squeezed, 1, Dense::Args().WithActivation(Identity));
    
    // Loss
    auto loss = ops::MeanSquaredError(root, output, target);
    
    // Optimizer
    auto optimizer = ops::AdamOptimizer(root, 0.001);
    auto train_op = optimizer.Minimize(root, loss);
    
    // Note: This is a simplified example - actual implementation would include
    // data loading, training loop, and proper tensor manipulation
    
    std::cout << "RNN graph defined successfully" << std::endl;
    return 0;
  }`,
        timeComplexity: "O(t × n²) where t is sequence length and n is hidden size",
        spaceComplexity: "O(t × n)",
        author: {
          name: "Numerical Computing Team"
        },
        lastUpdated: "2023-03-05",
        notes: "For production C++ applications, consider using TensorFlow Serving or ONNX Runtime for deployed models. Manual implementation of RNN cells is complex and error-prone."
      }
    },

    prosCons: {
      strengths: [
        'Can handle variable-length sequences',
        'Shares parameters across time steps',
        'Maintains memory of previous inputs',
        'Well-suited for temporal and sequential data'
      ],
      weaknesses: [
        'Suffers from vanishing/exploding gradients',
        'Computationally intensive for long sequences',
        'Difficult to parallelize across time steps',
        'Limited context window for long-range dependencies'
      ]
    },
    
    hyperparameters: [
      {
        name: 'hidden_units',
        type: 'range',
        min: 16,
        max: 512,
        step: 16,
        default: 64,
        description: 'Number of hidden units in RNN layer'
      },
      {
        name: 'rnn_type',
        type: 'select',
        options: ['simple', 'lstm', 'gru'],
        default: 'lstm',
        description: 'Type of recurrent cell to use'
      },
      {
        name: 'sequence_length',
        type: 'range',
        min: 10,
        max: 1000,
        step: 10,
        default: 50,
        description: 'Length of input sequences'
      }
    ],
    
    useCases: [
      {
        title: 'Time Series Forecasting',
        description: 'Predicting future values based on historical time series data.',
        dataset: 'Air Passengers, Stock Prices'
      },
      {
        title: 'Natural Language Processing',
        description: 'Processing text data for tasks like machine translation, sentiment analysis, and text generation.',
        dataset: 'IMDB Reviews, Penn Treebank'
      },
      {
        title: 'Speech Recognition',
        description: 'Converting spoken language into text by processing audio sequences.',
        dataset: 'LibriSpeech, Common Voice'
      }
    ],
    
    comparisons: [
      {
        algorithm: 'Transformers',
        comparison: 'RNNs process sequences sequentially while Transformers use self-attention to process all positions in parallel, with Transformers generally outperforming RNNs on long sequences but requiring more memory'
      },
      {
        algorithm: 'CNNs for Sequences',
        comparison: '1D CNNs can process sequences with dilated convolutions but lack explicit memory mechanisms, making RNNs better for tasks requiring long-term dependencies'
      }
    ],
    
    quiz: [
      {
        question: 'What problem do LSTM and GRU architectures primarily address?',
        options: [
          'Overfitting in neural networks',
          'Vanishing gradient problem in RNNs',
          'Computational complexity of deep networks',
          'Feature extraction from raw data'
        ],
        correct: 1,
        explanation: 'LSTM and GRU were designed with gating mechanisms to better preserve gradient flow and capture long-range dependencies, addressing the vanishing gradient problem in vanilla RNNs.'
      },
      {
        question: 'What is the key characteristic that distinguishes RNNs from feedforward networks?',
        options: [
          'RNNs have more layers',
          'RNNs have recurrent connections that form cycles',
          'RNNs use different activation functions',
          'RNNs require less data for training'
        ],
        correct: 1,
        explanation: 'The defining characteristic of RNNs is their recurrent connections that allow information to persist across time steps, creating an internal memory of previous inputs.'
      }
    ],
    
    projects: [
      {
        title: 'Text Generation with RNNs',
        description: 'Build a character-level RNN to generate text in the style of a given corpus.',
        steps: [
          'Preprocess text data and create character mappings',
          'Design RNN architecture with embedding and recurrent layers',
          'Train the model to predict next characters',
          'Generate new text by sampling from model predictions',
          'Experiment with temperature parameter to control randomness'
        ],
        difficulty: 'advanced',
        xp: 500
      }
    ]
  },

  // Multilayer Perceptron (MLP) AlgoData
  {
    id: 'mlp',
    title: 'Multilayer Perceptron (MLP)',
    category: 'deep',
    difficulty: 'beginner',
    tags: ['Neural Networks', 'Deep Learning', 'Supervised'],
    description: 'A class of feedforward artificial neural network that consists of at least three layers of nodes: input, hidden, and output layers.',
    icon: 'network-wired',
    lastUpdated: '2023-09-20',
    popularity: 0.92,
    
    concept: {
      overview: 'Multilayer Perceptrons are feedforward neural networks with one or more hidden layers between the input and output layers. They can learn non-linear relationships through activation functions and are universal function approximators.',
      analogy: 'Like a team of specialists where each layer extracts increasingly complex features from the input, similar to how our visual system processes images from edges to objects.',
      history: 'First proposed by Frank Rosenblatt in 1958 as perceptrons. The backpropagation algorithm was developed in the 1970s and popularized in the 1980s, enabling training of multi-layer networks.',
      mathematicalFormulation: {
        forwardPropagation: {
          equation: 'a⁽ˡ⁾ = σ(z⁽ˡ⁾) = σ(W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾)',
          description: 'Calculation of activations at each layer'
        },
        backpropagation: {
          equation: 'δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀδ⁽ˡ⁺¹⁾ ⊙ σ′(z⁽ˡ⁾)',
          description: 'Error calculation propagating backwards through the network'
        },
        activationFunctions: [
          {
            name: 'Sigmoid',
            formula: 'σ(x) = 1 / (1 + e⁻ˣ)',
            range: '(0, 1)'
          },
          {
            name: 'ReLU',
            formula: 'f(x) = max(0, x)',
            range: '[0, ∞)'
          },
          {
            name: 'Tanh',
            formula: 'tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)',
            range: '(-1, 1)'
          }
        ]
      },
      assumptions: [
        'The relationship between inputs and outputs can be modeled by a function',
        'There is enough data to learn the complex relationships',
        'Features are relevant to the prediction task'
      ]
    },
    
    visualization: {
      visualizerKey: 'mlp',
      defaultType: 'network-architecture',
      description: 'Interactive visualization of Multilayer Perceptron. Explore how information flows through the network and how weights are updated during training.',
      instructions: [
        'Adjust network architecture (layers, neurons)',
        'Change activation functions to see their effects',
        'Observe forward and backward propagation in real-time',
        'Monitor loss and accuracy during training'
      ],
      showTypeSelector: true,
      types: [
        { 
          value: 'network-architecture', 
          label: 'Network Architecture', 
          description: 'Visual representation of the neural network structure',
          default: true 
        },
        { 
          value: 'forward-propagation', 
          label: 'Forward Propagation',
          description: 'Show how inputs are transformed through layers'
        },
        { 
          value: 'backward-propagation', 
          label: 'Backpropagation',
          description: 'Visualize gradient flow during training'
        },
        { 
          value: 'decision-boundary', 
          label: 'Decision Boundary',
          description: 'See how the network separates classes'
        },
        { 
          value: 'all', 
          label: 'Complete View',
          description: 'All visualization elements together'
        }
      ],
      parameters: {
        n_samples: 200,
        n_classes: 3,
        n_layers: 2,
        n_neurons: 5,
        activation: 'relu',
        learning_rate: 0.01,
        epochs: 100,
        batch_size: 32,
        show_architecture: true,
        show_activations: false,
        show_gradients: false,
        show_loss: true,
        animation_duration: 3000,
        interactive: true
      },
      performanceTips: [
        'Deep networks with many layers may render slowly',
        'Gradient visualization is computationally intensive',
        'ReLU activation typically trains faster than sigmoid/tanh'
      ]
    },
    
    implementations: {
      python: {
        version: "3.9",
        libraries: ['scikit-learn@1.0.2', 'tensorflow@2.8.0', 'keras@2.8.0'],
        code: `# Python implementation using Keras/TensorFlow
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense
  from sklearn.datasets import make_classification
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler

  # Generate sample data
  X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

  # Split and scale data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  # Create MLP model
  model = Sequential([
      Dense(64, activation='relu', input_shape=(20,)),
      Dense(32, activation='relu'),
      Dense(1, activation='sigmoid')
  ])

  # Compile model
  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

  # Train model
  history = model.fit(X_train, y_train, 
                      epochs=50, 
                      batch_size=32,
                      validation_split=0.2,
                      verbose=1)

  # Evaluate
  test_loss, test_acc = model.evaluate(X_test, y_test)
  print(f"Test accuracy: {test_acc:.4f}")`,
        timeComplexity: "O(n × l × nₗ × nₗ₊₁) per epoch where l is layers, nₗ is neurons in layer l",
        spaceComplexity: "O(n × ∑nₗ) for storing activations and gradients",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For better performance, consider using GPU-accelerated TensorFlow. For large datasets, use batch normalization and dropout to improve generalization."
      },
      r: {
        version: "4.2",
        libraries: ['keras@2.9.0', 'tensorflow@2.9.0'],
        code: `# R implementation using Keras
  library(keras)
  library(tensorflow)

  # Generate sample data
  set.seed(42)
  data <- matrix(rnorm(1000 * 20), nrow = 1000, ncol = 20)
  labels <- ifelse(rowSums(data) > 0, 1, 0)

  # Split data
  train_idx <- sample(1:1000, 800)
  train_data <- data[train_idx, ]
  test_data <- data[-train_idx, ]
  train_labels <- labels[train_idx]
  test_labels <- labels[-train_idx]

  # Create MLP model
  model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = 'relu', input_shape = c(20)) %>%
    layer_dense(units = 32, activation = 'relu') %>%
    layer_dense(units = 1, activation = 'sigmoid')

  # Compile model
  model %>% compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = c('accuracy')
  )

  # Train model
  history <- model %>% fit(
    train_data, train_labels,
    epochs = 50,
    batch_size = 32,
    validation_split = 0.2,
    verbose = 1
  )

  # Evaluate
  metrics <- model %>% evaluate(test_data, test_labels)
  print(paste("Test accuracy:", metrics$accuracy))`,
        timeComplexity: "O(n × l × nₗ × nₗ₊₁) per epoch",
        spaceComplexity: "O(n × ∑nₗ)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "Make sure to install Python and TensorFlow before using the keras R package. For Windows users, consider using the install_keras() function for easy setup."
      },
      javascript: {
        version: "1.7.0",
        libraries: ['tensorflow.js@3.18.0'],
        code: `// JavaScript implementation using TensorFlow.js
  import * as tf from '@tensorflow/tfjs';

  // Generate sample data
  const xs = tf.randomNormal([1000, 20]);
  const ys = tf.tensor1d(Array.from({length: 1000}, 
    (_, i) => xs.arraySync()[i].reduce((a, b) => a + b, 0) > 0 ? 1 : 0));

  // Create MLP model
  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: 64,
    activation: 'relu',
    inputShape: [20]
  }));
  model.add(tf.layers.dense({
    units: 32,
    activation: 'relu'
  }));
  model.add(tf.layers.dense({
    units: 1,
    activation: 'sigmoid'
  }));

  // Compile model
  model.compile({
    optimizer: 'adam',
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });

  // Train model
  await model.fit(xs, ys, {
    epochs: 50,
    batchSize: 32,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(\`Epoch \${epoch}: loss = \${logs.loss}, accuracy = \${logs.acc}\`);
      }
    }
  });

  // Predict
  const sample = tf.randomNormal([1, 20]);
  const prediction = model.predict(sample);
  console.log('Prediction:', prediction.arraySync());`,
        timeComplexity: "O(n × l × nₗ × nₗ₊₁) per epoch",
        spaceComplexity: "O(n × ∑nₗ)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For better performance in the browser, use WebGL backend. For Node.js, use '@tensorflow/tfjs-node' which has native bindings."
      },
      cpp: {
        version: "17",
        libraries: ['mlpack@3.4.2', 'armadillo@11.2.1'],
        code: `// C++ implementation using mlpack
  #include <mlpack.hpp>
  #include <armadillo>
  #include <iostream>

  using namespace mlpack;
  using namespace arma;
  using namespace std;

  int main() {
    // Sample data (features and labels)
    mat features = randu<mat>(20, 1000);
    rowvec labels = sum(features, 0) > 10;  // Simple threshold function

    // Split data (80% train, 20% test)
    mat trainFeatures = features.cols(0, 799);
    mat testFeatures = features.cols(800, 999);
    rowvec trainLabels = labels.cols(0, 799);
    rowvec testLabels = labels.cols(800, 999);

    // Initialize MLP model
    FFN<NegativeLogLikelihood<>, RandomInitialization> model;
    
    // Add layers
    model.Add<Linear<>>(20, 64);  // Input to first hidden
    model.Add<ReLULayer<>>();
    model.Add<Linear<>>(64, 32);  // First hidden to second hidden
    model.Add<ReLULayer<>>();
    model.Add<Linear<>>(32, 1);   // Second hidden to output
    model.Add<SigmoidLayer<>>();

    // Set training parameters
    ens::Adam optimizer(0.01, 32, 0.9, 0.999, 1e-8, 50, 800, 1e-8, true);
    
    // Train model
    model.Train(trainFeatures, trainLabels, optimizer);

    // Predict
    mat predictions;
    model.Predict(testFeatures, predictions);
    
    // Calculate accuracy
    uvec correct = find((predictions > 0.5) == testLabels);
    double accuracy = correct.n_elem / (double)testLabels.n_elem;
    cout << "Test accuracy: " << accuracy << endl;

    return 0;
  }`,
        timeComplexity: "O(n × l × nₗ × nₗ₊₁) per epoch",
        spaceComplexity: "O(n × ∑nₗ)",
        author: {
          name: "Numerical Computing Team"
        },
        lastUpdated: "2023-03-05",
        notes: "mlpack provides efficient neural network implementations. For very large networks, consider using GPU-accelerated libraries like ArrayFire with mlpack."
      }
    },

    prosCons: {
      strengths: [
        'Can approximate any continuous function (universal approximation theorem)',
        'Can learn complex non-linear relationships',
        'Can learn from large amounts of data',
        'Works with various data types (with appropriate preprocessing)'
      ],
      weaknesses: [
        'Black box model (hard to interpret)',
        'Prone to overfitting without regularization',
        'Sensitive to feature scaling',
        'Computationally expensive to train'
      ]
    },
    
    hyperparameters: [
      {
        name: 'hidden_layer_sizes',
        type: 'text',
        default: '100,50',
        description: 'The number of neurons in each hidden layer, specified as a tuple'
      },
      {
        name: 'activation',
        type: 'select',
        options: ['relu', 'tanh', 'logistic'],
        default: 'relu',
        description: 'Activation function for the hidden layers'
      },
      {
        name: 'learning_rate',
        type: 'select',
        options: ['constant', 'invscaling', 'adaptive'],
        default: 'constant',
        description: 'Learning rate schedule for weight updates'
      },
      {
        name: 'alpha',
        type: 'range',
        min: 0.0001,
        max: 0.1,
        step: 0.0001,
        default: 0.0001,
        description: 'L2 regularization term'
      }
    ],
    
    useCases: [
      {
        title: 'Image Classification',
        description: 'Classifying images into categories using MLPs as the final classification layers after feature extraction.',
        dataset: 'MNIST, CIFAR-10'
      },
      {
        title: 'Fraud Detection',
        description: 'Identifying fraudulent transactions based on patterns in transaction data.',
        dataset: 'Credit Card Fraud Detection'
      }
    ],
    
    comparisons: [
      {
        algorithm: 'Decision Trees',
        comparison: 'MLPs can model more complex relationships but are less interpretable than decision trees'
      },
      {
        algorithm: 'Convolutional Neural Networks',
        comparison: 'MLPs treat input as flat vectors while CNNs preserve spatial structure and are better for image data'
      }
    ],
    
    quiz: [
      {
        question: 'What is the universal approximation theorem?',
        options: [
          'MLPs can approximate any continuous function given enough hidden units',
          'MLPs always converge to the global optimum',
          'MLPs are the most efficient neural network architecture',
          'MLPs can solve any machine learning problem'
        ],
        correct: 0,
        explanation: 'The universal approximation theorem states that a feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function.'
      },
      {
        question: 'Which activation function is most commonly used in hidden layers of modern MLPs?',
        options: [
          'Sigmoid',
          'Tanh',
          'ReLU',
          'Linear'
        ],
        correct: 2,
        explanation: 'ReLU (Rectified Linear Unit) is most commonly used because it helps mitigate the vanishing gradient problem and accelerates convergence.'
      }
    ],
    
    projects: [
      {
        title: 'Handwritten Digit Recognition',
        description: 'Build an MLP to classify handwritten digits from the MNIST dataset.',
        steps: [
          'Load and preprocess the MNIST dataset',
          'Design the network architecture',
          'Train the MLP with appropriate hyperparameters',
          'Evaluate performance on the test set',
          'Visualize misclassified examples'
        ],
        difficulty: 'intermediate',
        xp: 400
      }
    ]
  },

  // Transformer AlgoData
  {
    id: 'transformer',
    title: 'Transformer',
    category: 'deep',
    difficulty: 'advanced',
    tags: ['Deep Learning', 'NLP', 'Computer Vision', 'Supervised'],
    description: 'A deep learning model that utilizes self-attention mechanisms to process sequential data, revolutionizing natural language processing and other domains.',
    icon: 'language',
    lastUpdated: '2023-10-05',
    popularity: 0.98,
    
    concept: {
      overview: 'The Transformer is a neural network architecture based on self-attention mechanisms, dispensing with recurrence and convolutions entirely. It allows for significantly more parallelization than previous sequence-to-sequence models.',
      analogy: 'Like a team of experts where each member can directly attend to any part of the input sequence, rather than processing information sequentially.',
      history: 'Introduced in the 2017 paper "Attention Is All You Need" by Vaswani et al. at Google. Revolutionized NLP and led to models like BERT, GPT, and T5.',
      mathematicalFormulation: {
        attentionMechanism: {
          equation: 'Attention(Q, K, V) = softmax(QKᵀ/√dₖ)V',
          description: 'Scaled dot-product attention mechanism'
        },
        multiHeadAttention: {
          equation: 'MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)Wᵒ',
          description: 'Multiple attention heads allow the model to focus on different representation subspaces'
        },
        positionalEncoding: {
          equation: 'PE(pos, 2i) = sin(pos/10000^(2i/d_model))',
          equation2: 'PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))',
          description: 'Sinusoidal encoding to give positional information to the model'
        }
      },
      keyComponents: [
        'Self-Attention Mechanism',
        'Multi-Head Attention',
        'Positional Encoding',
        'Feed Forward Networks',
        'Layer Normalization',
        'Residual Connections'
      ]
    },
    
    visualization: {
      visualizerKey: 'transformer',
      defaultType: 'architecture',
      description: 'Interactive visualization of Transformer architecture. Explore how self-attention works and how information flows through the encoder-decoder structure.',
      instructions: [
        'Explore the encoder and decoder components',
        'Observe how attention weights change for different inputs',
        'Track how information flows through multiple layers',
        'Examine positional encoding patterns'
      ],
      showTypeSelector: true,
      types: [
        { 
          value: 'architecture', 
          label: 'Architecture Overview', 
          description: 'High-level view of encoder-decoder structure',
          default: true 
        },
        { 
          value: 'attention', 
          label: 'Attention Patterns',
          description: 'Visualize attention weights between tokens'
        },
        { 
          value: 'encoder-detail', 
          label: 'Encoder Details',
          description: 'Focus on encoder components and operations'
        },
        { 
          value: 'decoder-detail', 
          label: 'Decoder Details',
          description: 'Focus on decoder components and operations'
        },
        { 
          value: 'all', 
          label: 'Complete View',
          description: 'All visualization elements together'
        }
      ],
      parameters: {
        n_layers: 2,
        n_heads: 4,
        d_model: 64,
        sequence_length: 10,
        show_attention: true,
        show_encoder: true,
        show_decoder: true,
        show_embeddings: false,
        animation_duration: 4000,
        interactive: true
      },
      performanceTips: [
        'Visualizing many layers or long sequences may impact performance',
        'Attention visualization is most informative for short sequences',
        'The complete view provides the most comprehensive understanding'
      ]
    },
    
    implementations: {
      python: {
        version: "3.9",
        libraries: ['torch@1.13.0', 'transformers@4.24.0', 'numpy@1.21.5'],
        code: `# Python implementation using PyTorch and Hugging Face Transformers
  import torch
  import torch.nn as nn
  from transformers import AutoTokenizer, AutoModel

  # Load pre-trained transformer model and tokenizer
  model_name = "bert-base-uncased"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModel.from_pretrained(model_name)

  # Sample text
  text = "The transformer model revolutionized natural language processing."

  # Tokenize input
  inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

  # Forward pass through model
  with torch.no_grad():
      outputs = model(**inputs)

  # Get embeddings
  last_hidden_states = outputs.last_hidden_state
  print(f"Input shape: {inputs['input_ids'].shape}")
  print(f"Output shape: {last_hidden_states.shape}")

  # Alternatively, implement a simple transformer from scratch
  class SimpleTransformer(nn.Module):
      def __init__(self, vocab_size, d_model, nhead, num_layers):
          super(SimpleTransformer, self).__init__()
          self.embedding = nn.Embedding(vocab_size, d_model)
          self.transformer = nn.Transformer(
              d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
              num_decoder_layers=num_layers
          )
          self.fc = nn.Linear(d_model, vocab_size)
          
      def forward(self, src, tgt):
          src = self.embedding(src)
          tgt = self.embedding(tgt)
          output = self.transformer(src, tgt)
          return self.fc(output)

  # Example usage of custom transformer
  vocab_size = 1000
  d_model = 512
  nhead = 8
  num_layers = 3

  custom_transformer = SimpleTransformer(vocab_size, d_model, nhead, num_layers)
  print("Custom transformer created successfully")`,
        timeComplexity: "O(n² × d) for self-attention where n is sequence length, d is model dimension",
        spaceComplexity: "O(n² + n × d) for storing attention weights and activations",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For training from scratch, consider using a smaller model first. For most applications, fine-tuning pre-trained models is more efficient than training from scratch."
      },
      javascript: {
        version: "1.7.0",
        libraries: ['@tensorflow/tfjs@3.18.0', '@tensorflow-models/universal-sentence-encoder@1.3.3'],
        code: `// JavaScript implementation using TensorFlow.js
  import * as tf from '@tensorflow/tfjs';
  import * as use from '@tensorflow-models/universal-sentence-encoder';

  // Load the Universal Sentence Encoder (Transformer-based)
  const loadModel = async () => {
    const model = await use.load();
    return model;
  };

  // Example usage
  const runExample = async () => {
    const model = await loadModel();
    
    // Sample sentences
    const sentences = [
      'Hello, how are you?',
      'The transformer model is powerful.',
      'Natural language processing has evolved significantly.'
    ];
    
    // Generate embeddings
    const embeddings = await model.embed(sentences);
    console.log('Embeddings shape:', embeddings.shape);
    
    // Calculate similarity
    const similarityMatrix = tf.matMul(embeddings, embeddings, false, true);
    console.log('Similarity matrix:');
    similarityMatrix.array().then(matrix => console.log(matrix));
  };

  runExample().catch(console.error);

  // Simple attention implementation
  function scaledDotProductAttention(q, k, v, mask = null) {
    const dk = k.shape[k.shape.length - 1];
    const qk = tf.matMul(q, k.transpose([0, 2, 1]));
    const scaled = tf.div(qk, tf.sqrt(dk));
    
    if (mask) {
      scaled.add(tf.mul(mask, -1e9));
    }
    
    const weights = tf.softmax(scaled, -1);
    return tf.matMul(weights, v);
  }`,
        timeComplexity: "O(n² × d)",
        spaceComplexity: "O(n² + n × d)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For production applications, consider using Web Workers to avoid blocking the main thread. The Universal Sentence Encoder provides a good balance between performance and accuracy."
      },
      r: {
        version: "4.2",
        libraries: ['torch@0.9.0', 'huggingfacehub@0.1.0'],
        code: `# R implementation using torch and huggingfacehub
  library(torch)
  library(huggingfacehub)

  # Note: This is a conceptual example as R support for transformers is limited
  # Typically, you would use Python for transformer models

  # Sample text
  text <- "The transformer architecture has changed NLP forever."

  # In practice, you would typically:
  # 1. Use the reticulate package to call Python transformers
  # 2. Use an API service for transformer models
  # 3. Use specialized R packages like text for some transformer functionality

  # Example using the text package (if installed)
  if (require(text)) {
    # Load a pre-trained transformer model
    # Note: This downloads the model on first run
    model <- textEmbed(text, model = "bert-base-uncased")
    print(dim(model$tokens))
  }

  # Simple attention mechanism implementation in R/torch
  simple_attention <- function(query, key, value) {
    d_k <- dim(key)[length(dim(key))]
    scores <- torch_matmul(query, key$transpose(-2, -1)) / sqrt(d_k)
    weights <- nnf_softmax(scores, dim = -1)
    return(torch_matmul(weights, value))
  }

  # Example tensors
  q <- torch_randn(c(1, 5, 64))  # (batch, seq_len, d_model)
  k <- torch_randn(c(1, 5, 64))
  v <- torch_randn(c(1, 5, 64))

  # Apply attention
  output <- simple_attention(q, k, v)
  print(dim(output))`,
        timeComplexity: "O(n² × d)",
        spaceComplexity: "O(n² + n × d)",
        author: {
          name: "Roubhi Zakarya",
          link: "https://www.linkedin.com/in/zakaryaroubhi/"
        },
        lastUpdated: "2025-08-29",
        notes: "For serious transformer work in R, consider using the reticulate package to interface with Python libraries like transformers and torch."
      },
      cpp: {
        version: "17",
        libraries: ['torch@1.13.0', 'pybind11@2.10.0'],
        code: `// C++ implementation using LibTorch
  #include <torch/torch.h>
  #include <iostream>

  // Simple transformer implementation using LibTorch
  struct TransformerImpl : torch::nn::Module {
    TransformerImpl(int64_t d_model, int64_t nhead, int64_t num_layers) {
      // Embedding layer
      embedding = register_module("embedding", 
                  torch::nn::Embedding(1000, d_model));
      
      // Transformer
      transformer = register_module("transformer",
                    torch::nn::Transformer(torch::nn::TransformerOptions()
                      .d_model(d_model)
                      .nhead(nhead)
                      .num_encoder_layers(num_layers)
                      .num_decoder_layers(num_layers)));
      
      // Final linear layer
      fc = register_module("fc", torch::nn::Linear(d_model, 1000));
    }
    
    torch::Tensor forward(torch::Tensor src, torch::Tensor tgt) {
      src = embedding(src);
      tgt = embedding(tgt);
      
      auto output = transformer(src, tgt);
      return fc(output);
    }
    
    torch::nn::Embedding embedding{nullptr};
    torch::nn::Transformer transformer{nullptr};
    torch::nn::Linear fc{nullptr};
  };

  TORCH_MODULE(Transformer);

  int main() {
    // Model parameters
    int64_t d_model = 512;
    int64_t nhead = 8;
    int64_t num_layers = 3;
    
    // Create model
    Transformer model(d_model, nhead, num_layers);
    
    // Sample input
    auto src = torch::randint(0, 1000, {10, 32});  // (seq_len, batch_size)
    auto tgt = torch::randint(0, 1000, {10, 32});
    
    // Forward pass
    auto output = model(src, tgt);
    std::cout << "Output shape: ";
    for (auto dim : output.sizes()) {
      std::cout << dim << " ";
    }
    std::cout << std::endl;
    
    return 0;
  }`,
        timeComplexity: "O(n² × d)",
        spaceComplexity: "O(n² + n × d)",
        author: {
          name: "Numerical Computing Team"
        },
        lastUpdated: "2023-03-05",
        notes: "LibTorch provides a C++ API for PyTorch. For maximum performance, consider using TensorRT or ONNX Runtime for deployment."
      }
    },

    prosCons: {
      strengths: [
        'Superior parallelization compared to RNNs',
        'Captures long-range dependencies effectively',
        'State-of-the-art performance on many NLP tasks',
        'Flexible architecture applicable to various domains'
      ],
      weaknesses: [
        'Quadratic memory complexity with sequence length',
        'Computationally expensive for long sequences',
        'Large model sizes require significant resources',
        'Training requires massive amounts of data'
      ]
    },
    
    hyperparameters: [
      {
        name: 'd_model',
        type: 'range',
        min: 64,
        max: 1024,
        step: 64,
        default: 512,
        description: 'Dimension of the model (embedding size)'
      },
      {
        name: 'nhead',
        type: 'range',
        min: 1,
        max: 16,
        step: 1,
        default: 8,
        description: 'Number of attention heads'
      },
      {
        name: 'num_layers',
        type: 'range',
        min: 1,
        max: 12,
        step: 1,
        default: 6,
        description: 'Number of encoder/decoder layers'
      },
      {
        name: 'dim_feedforward',
        type: 'range',
        min: 128,
        max: 4096,
        step: 128,
        default: 2048,
        description: 'Dimension of the feedforward network'
      }
    ],
    
    useCases: [
      {
        title: 'Machine Translation',
        description: 'Translating text between languages using sequence-to-sequence transformer models.',
        dataset: 'WMT, OPUS'
      },
      {
        title: 'Text Generation',
        description: 'Generating coherent and contextually relevant text using decoder-only transformers.',
        dataset: 'Various text corpora'
      }
    ],
    
    comparisons: [
      {
        algorithm: 'RNN/LSTM',
        comparison: 'Transformers handle long-range dependencies better and allow more parallelization, but have higher memory requirements'
      },
      {
        algorithm: 'CNN',
        comparison: 'Transformers can capture global context immediately while CNNs build context gradually through layers'
      }
    ],
    
    quiz: [
      {
        question: 'What is the key innovation of the Transformer architecture?',
        options: [
          'Convolutional layers',
          'Recurrent connections',
          'Self-attention mechanism',
          'Pooling operations'
        ],
        correct: 2,
        explanation: 'The self-attention mechanism is the key innovation that allows Transformers to process all tokens in parallel and capture long-range dependencies effectively.'
      },
      {
        question: 'What is the computational complexity of self-attention?',
        options: [
          'O(n)',
          'O(n log n)',
          'O(n²)',
          'O(n³)'
        ],
        correct: 2,
        explanation: 'Self-attention has O(n²) complexity where n is the sequence length, due to the attention matrix calculation between all pairs of tokens.'
      }
    ],
    
    projects: [
      {
        title: 'Text Summarization',
        description: 'Build a transformer-based model to generate summaries of long documents.',
        steps: [
          'Preprocess and tokenize text data',
          'Fine-tune a pre-trained transformer model (e.g., T5, BART)',
          'Implement beam search for text generation',
          'Evaluate using ROUGE metrics',
          'Create a web interface for the summarizer'
        ],
        difficulty: 'advanced',
        xp: 600
      }
    ]
  },
];

// Map algorithms by ID for visualizations.js
window.algorithmData = {};
ALGORITHMS.forEach(alg => {
  window.algorithmData[alg.id] = alg;
});


