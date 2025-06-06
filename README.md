# XGBlog Exercise: Machine Learning

**Article Link**: (March 2025)  
[(Dealing With Missing Values, Part 1.)](https://www.xgblog.ai/p/dealing-with-missing-values-part)

**Topic**:  
> No real world data collection process is perfect, and we are often left with all sorts of noise in our dataset: incorrectly recorded values, non-recorded values, corruption of data, etc. If we are able to spot all those irregular points, oftentimes the best we can do is treat them as missing values. Missing values are the fact of life if you work in data science, machine learning, or any other field that relies on the real-world data. Most of us hardly give those data points much thought, and when we do we rely on many ready-made tools, algorithms, or rules of thumb to deal with them. However, to do them proper justice you sometimes need to dig deeper, and make a judicious choice of what to do with them. And what you end up doing with them, like in many other circumstances in data science, can be boiled down to the trusted old phrase of “it depends”. Missing data can significantly impact the results of analyses and models, potentially leading to biased or misleading outcomes. -Bojan Tunguz  


**Article Link**: (June 2025)  
[(Dealing With Missing Values, Part 2.)](https://www.xgblog.ai/p/dealing-with-missing-values-part-951)  

**Topic**:  
> Multivariate Imputation by Chained Equations (MICE) is a sophisticated approach for handling missing data, especially effective when the missing values follow a complex pattern under the Missing At Random (MAR) assumption. Unlike simpler methods that handle each feature independently, MICE iteratively models each feature based on the others, capturing the inherent relationships within your dataset.

> The MICE process works by first filling missing values with initial estimates, often simple ones like mean or median values. It then iteratively refines these estimates by modeling each feature with regression techniques, conditional on the others. This approach allows MICE to accurately preserve multivariate relationships and provide uncertainty estimates for the imputed values.

> However, MICE requires careful tuning. It involves deciding on the number of iterations to run and handling the computational complexity that arises from modeling each feature iteratively. -Bojan Tunguz