# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 1: Standardized Test Analysis

## Problem Statement

Both College Board and ACT Inc. have been making changes to their mission of making colleges accessible to a wide range of students and help make them ready for college.([*source*](https://www.number2.com/college-board-act/#Understanding_The_Relationship_Between_The_College_Board_and_The_ACT_Test)) The strategies include setting up school day exam, rolling out Assessment Suite, and turning tests online. This project aims to provide insights to both organisations and states/ districts educators by exploring the following :
    - the trend of particpation for both SAT and ACT from 2018 to 2019 
    - the correlation between participation rate and total score for both tests
    - the median of ACT & SAT test score for colleges of top 25 acceptance rate. 

**Problem Statement**: Which states shall each organisation focus to boost score performance while maintaining high participation rate so as to be effective in helping students?

## Background

The SAT and ACT are standardized tests that many colleges and universities in the United States require for their admissions process. This score is used along with other materials such as grade point average (GPA) and essay responses to determine whether or not a potential student will be accepted to the university.

The SAT has two sections of the test: Evidence-Based Reading and Writing and Math ([*source*](https://www.princetonreview.com/college/sat-sections)). The ACT has 4 sections: English, Mathematics, Reading, and Science, with an additional optional writing section ([*source*](https://www.act.org/content/act/en/products-and-services/the-act/scores/understanding-your-scores.html)). They have different score ranges, which you can read more about on their websites or additional outside sources (a quick Google search will help you understand the scores for each test):
* [SAT](https://collegereadiness.collegeboard.org/sat)
* [ACT](https://www.act.org/content/act/en.html) 

Standardized tests have long been a controversial topic for students, administrators, and legislators. Since the 1940's, an increasing number of colleges have been using scores from sudents' performances on tests like the SAT and the ACT as a measure for college readiness and aptitude ([*source*](https://www.minotdailynews.com/news/local-news/2017/04/a-brief-history-of-the-sat-and-act/)). Despite the fact that SAT will always be imperfect tool, it provides a more level playing field.[*source*](https://www.newyorker.com/news/annals-of-education/how-the-pandemic-remade-the-sat) 

Both organisations have been making changes to their mission and trying to meet the various needs from different parties. These include students, their families, college counselors, K-12 educators, researchers, policymakers, universities, colleges, athletic conferences, scholarship oragnisation and etc. The College Board has a new adversity score which responds to a growing trend among colleges to attract and enroll first-generation students and those with few resources. Similarly, ACT has been working on an approach to bolster student diversity through the ACT Holistic Framework.

### Datasets chosen 

* [`act_2018.csv`](./data/act_2018.csv): 2018 ACT Scores by State
* [`act_2019.csv`](./data/act_2019.csv): 2019 ACT Scores by State
* [`sat_2018.csv`](./data/sat_2018.csv): 2018 SAT Scores by State
* [`sat_2019.csv`](./data/sat_2019.csv): 2019 SAT Scores by State
* [`sat_2019_by_intended_college_major.csv`](./data/sat_2019_by_intended_college_major.csv): 2019 SAT Scores by Intended College Major
* [`sat_act_by_college.csv`](./data/sat_act_by_college.csv): Ranges of Accepted ACT & SAT Student Scores by Colleges

## Data Dictionary

|Feature|Type|Dataset|Description|
|---|---|---|---|
|**state**|*object*|final_1819|The state in the US and District of Columbia.| 
|**2018_sat_participation**|*float64*|final_1819|The participation rate of SAT test among high school seniors in a state for year 2018 (units percent to 2 decimal places 0.06 means 6%).| 
|**2018_sat_erbw**|*int64*|final_1819|The average SAT test score for subject "Evidence-Based Reading and Writing" in a state for year 2018 (min 200, max 800).|
|**2018_sat_math**|*int64*|final_1819|The average SAT test score for subject "Math" in a state for year 2018 (min 200, max 800).|
|**2018_sat_total**|*int64*|final_1819|The total of average SAT test score for both subjects in a state for year 2018 (min 400, max 1600).|
|**2018_act_participation**|*float64*|final_1819|The participation rate of ACT test among high school seniors in a state for year 2018 (units percent to 2 decimal places 0.41 means 41%).|
|**2018_act_composite**|*float64*|final_1819|The mean of average ACT test score for all four subjects "English", "Math", "Reading", & "Science" in a state for year 2018 (min 1.0, max 36.0).|
|**2019_sat_participation**|*float64*|final_1819|The participation rate of SAT test among high school seniors in a state for year 2019 (units percent to 2 decimal places 0.06 means 6%).| 
|**2019_sat_erbw**|*int64*|final_1819|The average SAT test score for subject "Evidence-Based Reading and Writing" in a state for year 2019 (min 200, max 800).|
|**2019_sat_math**|*int64*|final_1819|The average SAT test score for subject "Math" in a state for year 2019 (min 200, max 800).|
|**2019_sat_total**|*int64*|final_1819|The total of average SAT test score for both subjects in a state for year 2019 (min 400, max 1600).|
|**2019_act_participation**|*float64*|final_1819|The participation rate of ACT test among high school seniors in a state for year 2019 (units percent to 2 decimal placess 0.41 means 41%).|
|**2019_act_composite**|*float64*|final_1819|The mean of average ACT test score for all four subjects "English", "Math", "Reading", & "Science" in a state for year 2019 (min 1.0, max 36.0).|

|Feature|Type|Dataset|Description|
|---|---|---|---|
|**school**|*object*|sat_act_college|College or University name.| 
|**are_tests_optional**|*object*|sat_act_college|Yes or No statement regarding presence of test-optional policies at the school.|
|**last_class_year**|*object*|sat_act_college|The last application year the test-optional policy will be in place.|
|**policy_details**|*object*|sat_act_college|Details of the test optional policy.|
|**num_of_applicants**|*int64*|sat_act_college|Number of applicants to the school.|
|**acceptance_rate**|*float64*|sat_act_college|Percent (%) of students accepted among those who applied.|
|**sat_total_middle_range**|*object*|sat_act_college|Range of 25th and 75th percentile SAT scores accepted by the school.|
|**act_total_middle_range**|*object*|sat_act_college|Range of 25th and 75th percentile ACT scores accepted by the school.|
|**sat_25th_percentile**|*float64*|sat_act_college|25th percentile of school's accepted SAT scores.|
|**sat_75th_percentile**|*float64*|sat_act_college|75th percentile of school's accepted SAT scores.|
|**act_25th_percentile**|*float64*|sat_act_college|25th percentile of school's accepted ACT scores.|
|**act_75th_percentile**|*float64*|sat_act_college|75th percentile of school's accepted SAT scores.|

## Brief Summary
For both years 2018 and 2019, the ACT participation rate are distributed within a significantly higher range than that of the SAT participation rate. The right skewness of ACT boxplot diagram suggests stronger support for ACT test thoughout US.

We observe that ACT participation has a strong negative correlation with ACT composite scores. This means that states with lower ACT participation tend to have higher ACT scores, and vice versa for states with high ACT participation rates. The same for SAT participation rates, where there is an equally strong negative correlation between participation rates and SAT scores.

Total and test scores by subjects from 2018 are strongly correlated with the total and same subject test scores in 2019. This means that states are likely to continue doing well from year to year. This is unsurprising as states are unlikely to dramatically go up or down in test scores over a single year, due to policies and institutions (e.g. state education departments) that work to maintain consistent educational results year-on-year. Barring dramatic policy changes like switching from one test to another, states are likely to produce to the same results over the years.

The total test scores / participation rates for SAT is negatively correlated with that for ACT of the same year, and vice versa. It's pretty unusual for a student to take both tests in a year considering the time, effort and money factors. From another perspective, one of the possibility is self-selection bias and the effect may be larger than expected. Knowing that some states mandate students to take up one of the testsand students may want to sit for another in order to entera particular college. In addition, some students who want to take up both tests or take up the same test at different periods within the same year to increase their chances of obtaining a good score and make their studying more worth-while. 

## Conclusions/ Recommendations
When we look at performance across the standardized test in the 2018-19 school year, it offers a high-level performance summary. It’s impossible to infer whether shifts in performance are due to changes in the underlying ability of students or reflect shifts in the test-taking population. Both College Board and ACT Inc. have been finding ways to improve students' participation with the ultimate mission in mind through the years. Many controversial over these college admissions tests are being biased against low-income and non-white students, the reality is that these tests are the gatekeeper to selective colleges in the US. The evidence indicates that if taking these tests is voluntary, many talented, disadvantaged students will go undetected.

..To maximize the benefits of which both tests offered, I recommend one state for each test organization to further work on them. 

The state that SAT should help further is **West Virginia**, given that it has close to 100% participation but low mean total score. This value on year 2019 is much lower than the median of the highest acceptance college.  

While the state that ACT should help further is **Arizona**, given that it's favoring ACT test over SAT. The participation rate is increasing from year 2018 to 2019.

A good score can help students to achieve their academic goals, whether that means getting into a particular college or allocating with better placement or earning financial aid. While higher mean total score of SAT or ACT favors the admission, this is statistically significant to be factored in. With participation rate of a state negatively correlated to mean total score from that state, both tests should not focus too much on increasing the participation rate. 

My recommendation for the above-mentioned target audience is:
- help students apply for college
- work in tandem with state high schools and colleges to make sure students have reliable path
- partner with state boards of education to develop the standardized test
- continue offering it for free to qualified students
- market a reasonable ReTesting package
- continue School Day testing
- update prepping material/ asssessment suite regularly and inform students earlier


**To-Do:** *If you combine your problem statement, data dictionary, brief summary of your analysis, and conclusions/recommendations, you have an amazing README.md file that quickly aligns your audience to the contents of your project.* Don't forget to cite your data sources!
