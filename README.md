# nl4ds
Natural language interface for data science tasks

This toolset is initially developed for my colleagues at the Humanities Research Lab at NYU Shanghai. It has two simple functionalities: `select` subsets of data based on natural language queries and `describe` the data by printing summary statistics and plotting a suitable graph based on data type and distribution. Later, I added functionality and to `show_corr`elations between fields and packaged them into a script named ***eda_functions***. In the summer of 2021, I worked heavily with OCR and CompVision tasks, so I made two more toolsets to speed up my work, ***ocr_functions*** and ***map_functions***. Finally, I refactored an experimental code I used to test program synthesis on web scraping into a script named ***scraping_functions***.

The scripts have different use cases, but the design philosiphy is all about **allowing non-tech users write natural-language-like codes to perform data science tasks in their domain work.**

For example, if you want to select a subset of your data for further study, instead of learning and writing complex code like this:

`whole_population[(whole_population['Gender']=='Female') & (whole_population['Occupation'].fillna('').str.contains('keeper')) & (~whole_population['Occupation'].fillna('').str.contains('bookkeeper')) & (whole_population['Age']>=15) & (whole_population['Age']<64)]`

You can write something more natural and intuitive:

`select_data(whole_population, criteria = 'Gender is Female, Occupation contains keeper but not bookkeeper, Age is in [15, 64]')`

<br><br>

### *Re-searching* the Logic and Grammar of Data Wrangling

I haven't made any major update to the eda_functions since I first wrote them in the fall of 2020. Recently, I am re-introduced to the R programming language thanks to a course at Columbia called Exploratory Data Analysis and Visualization (EDAV). When using `dplyr` and `ggplot2`, I started to reflect on how I do data wrangling and EDA in Python. 

The more I think about it, it become clearer that syntactic differences, though important, are not the only reason that makes data wrangling hard. To make data wrangling **accessible** to domain experts and **efficient** for data practitioners, we need to have clear idea of the logic behind different actions and processes in data wrangling and then find the most intuitive syntax to map onto the logic. 

Below are some readings I gathered for analyzing the existing solutions. Since data wrangling can be done for many different purposes, I would focus on data wrangling for exploratory data analysis. For this reason, I also plan to read the literature on languages and grammar for visualization, and jointly searching for the logic and grammar suitable for the tasks.

**References for close reading:**

Dplyr [official page with cheatsheet](https://dplyr.tidyverse.org) 

Prof. Luke Tierney's [Dplyr tutorial](https://homepage.divms.uiowa.edu/~luke/classes/STAT4580/dplyr.html) 

Sharon Machlis's [list of R packages for data wrangling](https://www.computerworld.com/article/2921176/great-r-packages-for-data-import-wrangling-visualization.html)

Pandas official [user guides](https://pandas.pydata.org/docs/user_guide/index.html) 

Wes McKinney's ["10 things I hate about Pandas"](https://wesmckinney.com/blog/apache-arrow-pandas-internals/)

Software Engineering Daily's [Review of Python Data Wrangling Libraries 2020](https://softwareengineeringdaily.com/2020/08/26/in-with-the-new-python-plotting-and-data-wrangling-libraries/) 

SolutionsReview.com's [The 10 Best Data Wrangling Tools and Software for 2021](https://solutionsreview.com/data-integration/the-best-data-wrangling-tools-and-software/)