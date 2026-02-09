\documentclass[a4paper,fleqn]{cas-sc}

% Packages
\usepackage[authoryear,longnamesfirst]{natbib}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{float}
\usepackage{caption} 

\usepackage{titlesec}

% Increase spacing before and after subsections
\titlespacing*{\subsection}
  {0pt}        % left margin
  {1.5em}      % space before
  {1em}        % space after

\titlespacing*{\subsubsection}
  {0pt}
  {1.2em}
  {0.8em}


\begin{document}

\let\WriteBookmarks\relax
\def\floatpagepagefraction{1}
\def\textpagefraction{.001}

\shorttitle{Machine Learning for Insurance Customer Analytics}
\shortauthors{Jerono}

\title[mode = title]{Integrated Predictive Analytics for Customer Retention and Value Optimization in Automobile Insurance: A Machine Learning Approach}

\author{Valerie Jerono}
\ead{valerie.jerono@strathmore.edu}
\address[1]{MSc Data Science and Analytics, Strathmore University, Nairobi, Kenya}

\begin{abstract}
In automobile insurance markets, acquiring new customers costs five to twenty-five times more than retaining existing ones, making customer defection a critical profitability concern. This research develops and validates an integrated analytical framework comprising four interconnected predictive models, applied to 105,555 motor insurance contracts from a European provider spanning 2015 to 2018.

The framework integrates four analytical engines. First, retention analytics employs gradient boosting classification to forecast policy lapse with 71.5\% discrimination accuracy, identifying 25.9\% of policyholders as elevated renewal risks. Second, risk analytics achieves 92.3\% accuracy in predicting claims frequency, demonstrating machine learning's marked advantage over conventional statistical models for zero-inflated distributions. Third, lifetime value modeling through probabilistic cash flow methodology quantifies \texteuro{}25.8 million aggregate portfolio worth and reveals \texteuro{}483 systematic value disparities between distribution channels. Fourth, journey analytics enables strategic intervention through segment-based classification capturing value-risk evolution patterns.

The analytical approach implements gradient boosting with temporal validation using 80/20 training-test splits, addressing severe class imbalance through weight adjustments of 3.9:1 for retention modeling and 4.4:1 for claims prediction. Feature engineering produces fifteen composite indicators capturing temporal patterns, risk concentration metrics, and pricing adequacy measures. Empirical findings reveal critical vulnerability periods: policies aged one to three years exhibit 26.5\% lapse rates compared to 16.7\% for mature relationships exceeding ten years, representing 58\% elevated risk warranting specialized retention programs.

Distribution channel analysis demonstrates agent-sourced policyholders generate 752\% return on investment versus broker channel's 297\%, attributable to 60\% longer tenure (8.23 versus 4.84 years), 14\% reduced loss ratios (44.3\% versus 53.4\%), and 21\% lower attrition. Pricing evaluation exposes systematic underpricing affecting 14\% of contracts where premium levels fail to cover anticipated loss costs, with broker channel exhibiting 10\% adequacy gaps despite 8.5\% higher nominal premiums.

A Retrieval-Augmented Generation architecture operationalizes insights through natural language interfaces querying all four models concurrently, converting 53,502 customer profiles into conversationally accessible intelligence achieving 82\% production readiness with 24ms median query latency. Three-month pilot implementation with twenty agents managing 12,000 relationships validates measurable impact: 12.3\% attrition reduction, 35\% efficiency enhancement in risk identification, and \texteuro{}2.37 million projected annual value generation, producing 3,386\% first-year return on \texteuro{}70,000 implementation investment.

Interactive deployment: \url{https://automobilecustomerx.streamlit.app/}
\end{abstract}

\begin{keywords}
Customer churn prediction \sep Insurance analytics \sep Gradient boosting \sep Customer lifetime value \sep Risk-based pricing \sep RAG systems \sep Emerging markets \sep Machine learning
\end{keywords}

\maketitle

\section{Introduction}

Policyholder retention has emerged as a fundamental driver of insurance profitability globally. Empirical research establishes that incremental improvements of just 5\% in customer retention rates can amplify profits by 25\% to 95\%, demonstrating the asymmetric financial consequences of customer defection \citep{kumar2024retention}. This economic reality intensifies in emerging markets where insurance adoption remains structurally constrained and customer acquisition expenditures are disproportionately high relative to premium volumes.

Kenya's insurance penetration approximates 2.4\%, substantially below the global 7.0\% benchmark, implying that each lost policyholder represents not merely immediate premium erosion but forfeiture of considerable long-term customer lifetime value \citep{oecd2024insurance}. The challenge manifests acutely in automobile insurance, which constitutes Kenya's largest general insurance segment while experiencing severe structural pressures. Market fragmentation produces aggressive price competition among numerous insurers, generating compressed margins and pervasive underpricing practices \citep{wanjiru2023competitive}. These dynamics are amplified by elevated motor insurance fraud levels and escalating claim rejections that corrode trust and intensify policyholder dissatisfaction \citep{mwangi2024fraud}.

Within this environment, customer attrition materializes silently at renewal junctures, rendering retention simultaneously economically critical and operationally complex. Conventional customer management approaches have predominantly operated reactively: responding to churn after cancellation, to fraud after detection, and to profitability erosion after financial reporting reveals portfolio deterioration. This reactive orientation persists despite substantial advances in predictive analytics and machine learning enabling proactive intervention.

Analysis of 105,555 automobile insurance policies exposes four critical gaps in contemporary industry practice. Insurers lack systematic frameworks for forecasting customer departure prior to renewal dates, forfeiting opportunities for timely intervention. Claims prediction relies on traditional actuarial methodologies that fail to exploit behavioral data's predictive power. Customer lifetime value calculations, when executed, depend on static assumptions that ignore dynamic churn probabilities. Customer segmentation remains predominantly demographic rather than behavioral, preventing differentiated management aligned with value-risk trajectories.

This paper addresses these deficiencies by proposing and validating a comprehensive, integrated customer analytics framework specifically designed for automobile insurance in emerging markets. The framework unifies four interdependent predictive models: retention analytics for churn probability prediction, risk analytics for claims frequency forecasting, lifetime value calculation for long-term profitability quantification, and journey segmentation for differentiated management strategies.

\section{Literature Review}

\subsection{Customer Retention and Churn Prediction}

Customer defection constitutes a persistent profitability threat in competitive automobile insurance markets. Contemporary literature establishes that machine learning methodologies substantially surpass traditional statistical approaches in churn forecasting by capturing non-linear behavioral patterns and temporal dynamics.

\citet{zhang2023ensemble} demonstrated that ensemble learning techniques combining multiple gradient boosting algorithms attain superior predictive accuracy in insurance churn prediction, with discrimination accuracy exceeding 85\% in large-scale implementations. \citet{afriyie2024machine} showed through systematic comparison that tree-based models, particularly XGBoost and LightGBM, consistently outperform logistic regression and neural networks for insurance churn tasks due to native capability for handling class imbalance and capturing feature interactions automatically.

\citet{tharmarajan2024comparative} conducted comprehensive benchmarking across financial services datasets, confirming gradient boosting methods dominate churn prediction when appropriately tuned for imbalanced data. However, existing research predominantly treats churn prediction as an isolated task, rarely integrating churn risk with claims behavior or long-term customer value, thereby constraining strategic applicability.

\subsection{Claims Frequency, Severity, and Fraud Risk}

Precise prediction of claims frequency and severity represents fundamental requirements for underwriting discipline and profitability in automobile insurance. Recent advances demonstrate that gradient boosting algorithms have emerged as the dominant modeling paradigm for claims analytics due to superior performance on non-linear, zero-inflated insurance data.

\citet{richman2023ai} provided comprehensive evidence that artificial intelligence approaches, particularly gradient boosting and neural networks, substantially outperform traditional generalized linear models for both claims frequency and severity prediction across multiple insurance lines. \citet{avanzi2023boosting} demonstrated that tree-based boosting methods effectively handle zero-inflated characteristics of insurance claims data while maintaining interpretability through SHAP values and feature importance metrics.

\citet{henckaerts2022boosting} showed that incorporating spatial and temporal features into gradient boosting frameworks significantly improves predictive accuracy for automobile insurance claims. Parallel research on insurance fraud detection indicates ensemble classifiers achieve optimal balance between detection accuracy and false positive rates \citep{hilal2022financial}. Despite these advances, claims and fraud models typically develop independently of customer retention and value analytics, preventing insurers from understanding how risk dynamics interact with customer behavior.

\subsection{Customer Lifetime Value Estimation}

Customer Lifetime Value has become a central metric guiding retention investment and portfolio optimization in insurance. Recent studies emphasize that traditional actuarial approaches often fail to capture complex interactions between tenure, claims behavior, and policy characteristics.

\citet{chamberlain2024customer} demonstrated that machine learning regressors incorporating behavioral features significantly outperform linear models in predicting insurance lifetime value by modeling non-linear profitability drivers and customer interaction effects. \citet{liu2023predictive} showed that incorporating customer engagement metrics and digital behavioral data substantially improves lifetime value prediction accuracy in insurance contexts. \citet{kumar2024retention} provided empirical evidence that value-based segmentation enables more effective loyalty program design and targeted retention strategies, resulting in improved profitability and reduced marketing waste.

Nonetheless, most lifetime value studies rely on static assumptions and fail to incorporate dynamic churn probabilities or forward-looking risk estimates, limiting usefulness for proactive decision-making.

\subsection{Customer Segmentation and Journey Analysis}

Customer segmentation plays a critical role in enabling differentiated strategies across heterogeneous insurance portfolios. Behavioral segmentation approaches, particularly those based on Recency, Frequency, and Monetary variables, have proven more actionable than demographic segmentation in insurance contexts.

\citet{safari2024customer} demonstrated that combining these analytical frameworks with machine learning clustering algorithms produces meaningful insurance customer segments strongly correlated with profitability and engagement outcomes. \citet{alzahrani2023customer} showed that dynamic segmentation approaches recognizing customer journey stages enable more effective personalization strategies than static demographic categories.

Recent research increasingly frames segmentation as a dynamic customer journey problem, recognizing that customers migrate between value-risk states over time rather than remaining in static categories. However, most segmentation studies remain descriptive in nature and do not integrate predictive churn, claims, and lifetime value metrics, limiting ability to support proactive customer management and early intervention strategies.

\subsection{Operationalizing Analytics with Retrieval-Augmented Generation}

Despite significant advances in predictive modeling, operational adoption of analytics in insurance remains constrained by interpretability and accessibility barriers. Retrieval-Augmented Generation has recently emerged as a promising solution by combining large language models with domain-specific knowledge retrieval to deliver accurate, context-aware insights.

\citet{gao2023retrieval} provided comprehensive evidence that these architectures significantly reduce hallucinations and improve factual accuracy in knowledge-intensive domains compared to pure generative approaches. \citet{lewis2020retrieval} demonstrated that systems combining dense retrieval with generative language models achieve state-of-the-art performance on question-answering tasks requiring factual knowledge. \citet{zhao2024survey} conducted systematic analysis showing that such systems outperform traditional chatbots in financial services applications through enhanced accuracy and explainability.

These findings suggest Retrieval-Augmented Generation can bridge the gap between advanced analytics and frontline insurance decision-making. However, academic research has yet to systematically explore integration of these systems with multi-model customer analytics frameworks in insurance contexts.

\subsection{Research Gap and Objectives}

The reviewed literature demonstrates substantial progress in churn prediction, claims modeling, lifetime value estimation, customer segmentation, and decision support systems when considered in isolation. Nevertheless, a critical gap persists in the absence of integrated customer analytics frameworks tailored to emerging insurance markets, where low penetration rates, high fraud prevalence, intense price competition, and evolving regulatory environments create unique challenges.

This study addresses this gap by proposing and validating an integrated four-model framework combining retention analytics, risk analytics, value optimization, and journey segmentation with natural language interfaces, providing a unified and operationally actionable approach to automobile insurance analytics in emerging market contexts.

\section{Materials and Methods}

This study employs the Cross-Industry Standard Process for Data Mining framework \citep{schroder2021crisp}, encompassing six phases: business understanding, data understanding, data preparation, modeling, evaluation, and deployment.

\subsection{Data Source and Characteristics}

\subsubsection{Dataset Origin}

Administrative records were obtained from ICPSR's open repository, representing a European insurance company's non-life motor vehicle insurance portfolio spanning November 1, 2015 to December 1, 2018. The dataset comprises 105,555 policy transactions with thirty variables: twenty-nine features and one target variable indicating policy lapse. Despite European origins, portfolio dynamics reflect universal insurance challenges documented across markets \citep{richman2023ai}, making findings transferable to emerging markets characterized by competitive fragmentation, price sensitivity, and fraud pressures.

\subsubsection{Variable Taxonomy}

The dataset exhibits comprehensive coverage across critical insurance dimensions. Customer profile variables include seniority, measuring relationship duration with mean of 6.7 years (standard deviation 5.8, range 0 to 38 years), and policies in force, capturing cross-selling penetration with mean of 1.3 policies per customer.

Demographics encompass driver age (mean 47.9 years) and license years (mean 25.3 years), exhibiting strong correlation of $r = 0.89$, indicating limited independent information content. Policy details include premium with mean of \texteuro{}315.89 (standard deviation \texteuro{}201.45, range \texteuro{}23 to \texteuro{}8,945), distribution channel (54.9\% agent-sourced, 45.1\% broker-sourced), and payment frequency (68.1\% annual, 31.9\% semi-annual).

Geography distinguishes area as rural (72.6\%) or urban (27.4\%). Vehicle specifications comprise vehicle value with mean of \texteuro{}18,413, power with mean of 92.68 horsepower, and vehicle type distributed as passenger cars (79.3\%), vans (13.2\%), motorbikes (4.8\%), and agricultural vehicles (2.7\%).

Claims history features number of historical claims with mean of 2.75 claims and historical claims rate with mean of 0.28 claims per year. This historical claims rate, calculated as total claims divided by relationship duration, serves as the dominant predictor across models.

The target variable lapse exhibited 20.4\% incidence (21,548 of 105,555 policies), representing moderate class imbalance with 3.9:1 ratio of continuing to lapsing policies, suitable for gradient boosting with class weighting approaches \citep{avanzi2023boosting}. Claims incidence reached 18.6\% (19,646 claimants) with similar imbalance characteristics of 4.4:1 ratio. Among claimants, claim costs ranged from \texteuro{}23 to \texteuro{}34,890 with mean of \texteuro{}467 and median of \texteuro{}285, exhibiting substantial right skewness requiring logarithmic transformation for regression modeling.

\subsection{Data Preprocessing}

\subsubsection{Missing Value Treatment}

Overall dataset completeness reached 97.39\%, with missing values concentrated in three variables requiring theoretically appropriate imputation strategies based on missingness mechanisms \citep{little2019statistical}.

\textbf{Structurally Missing Data.} Lapse date exhibited 90,007 missing values representing 85.5\% missingness. In survival analysis terminology, these represent right-censored observations: policies that had not lapsed by the observation window end date. Following survival analysis best practices \citep{austin2022practical}, this variable was retained with the binary lapse indicator serving as the censoring variable. No imputation was applied, as missingness conveys meaningful information about policy continuation.

\textbf{Missing Completely at Random.} Vehicle length (10,347 missing values, 9.8\% missingness) and fuel type (1,794 missing, 1.7\%) exhibited random missingness patterns with no systematic relationship to observed variables, confirmed through Little's test: $\chi^2 = 23.4$, degrees of freedom equals 18, $p = 0.31$. Following comparative research demonstrating K-Nearest Neighbors superiority for such data \citep{bertsimas2023machine}, vehicle length employed K-Nearest Neighbors imputation with five neighbors and inverse distance weighting, preserving multivariate relationships. Fuel type utilized mode imputation within vehicle risk type segments, maintaining category distributions.

\textbf{Missing at Random.} Contract start date (60,375 missing, 57.2\%) and next renewal date (59,639 missing, 56.5\%) exhibited patterns where missingness probability depended on observed variables but not on the missing values themselves. Group-based imputation calculated contract start date from last renewal date minus seniority, leveraging the deterministic relationship between contract inception and renewal timing.

\subsubsection{Outlier Management}

Rather than removing outliers and losing legitimate extreme cases representing important business scenarios, this study employed winsorization to preserve record counts while constraining values to acceptable ranges \citep{hastie2020statistical}. Outliers were identified through interquartile range analysis and domain knowledge consultation, then capped at statistically determined thresholds:
\begin{equation}
x_{\text{capped}} = \min(x, U_p) \quad \text{and} \quad x_{\text{floored}} = \max(x, L_p)
\end{equation}
where $U_p = \text{quantile}(x, p_{\text{upper}})$ and $L_p = \text{quantile}(x, p_{\text{lower}})$.

Specifically, premium was capped at the 99th percentile of \texteuro{}1,200 (affecting 1,056 records), vehicle value at the 95th percentile of \texteuro{}45,000 (affecting 5,278 records), vehicle power at the 98th percentile of 200 horsepower (affecting 2,111 records), and claim costs at the 99th percentile of \texteuro{}8,500 (affecting 196 records).

\subsubsection{Feature Engineering}

Comprehensive feature engineering created fifteen composite variables capturing domain knowledge and theoretical relationships. Temporal features included vehicle age, calculated from contract start date and vehicle year, capturing depreciation effects on claims severity, and days to renewal, measuring difference between next renewal date and observation date, enabling renewal risk urgency assessment.

Risk concentration metrics comprised premium-to-value ratio, calculated as premium divided by vehicle value with mean of 0.0217, identifying underpriced policies at thresholds below 0.015, and binary indicator for claims history presence, with 65.3\% of policies showing prior claims.

Interaction features combined payment frequency with number of drivers, capturing commitment signals from annual payment choices, and distribution channel with geographic area, identifying channel-geographic interaction effects on retention. Standardization applied z-score normalization to continuous predictors ensuring comparable scales for tree-based algorithms, while categorical variables retained original encodings suitable for gradient boosting's native categorical handling.




% ============================================================================
% REPLACE YOUR ENTIRE EDA SECTION WITH THIS CODE
% This version uses minipage to force figures inline with text
% ============================================================================

\subsection{Exploratory Data Analysis}

This section presents three critical patterns discovered through comprehensive data exploration, each revealing actionable business insights that directly inform our modeling strategy and operational recommendations.

\subsubsection{The Lifecycle Valley of Death}

Analysis of customer tenure reveals a pronounced vulnerability window during the early relationship stages, as illustrated in Figure~\ref{fig:lifecycle}. Policies aged one to three years exhibit 26.5\% lapse rates, representing 58\% elevated risk compared to the portfolio average of 20.4\% and 59\% higher than mature relationships exceeding ten years (16.7\% lapse rate).

\begin{center}
\begin{minipage}{0.75\textwidth}
\centering
\includegraphics[width=\textwidth]{Lifecycle.png}
\captionof{figure}{Customer Lifecycle Vulnerability: Lapse rates by tenure years showing the pronounced "valley of death" during years 1--3. Mature relationships (10+ years) demonstrate substantially lower attrition.}
\label{fig:lifecycle}
\end{minipage}
\end{center}

This "valley of death" pattern indicates that the initial honeymoon period (year one with 11.2\% lapse) transitions into a critical decision point where customers actively re-evaluate their insurance relationship. The vulnerability persists through years three to five (24.9\% lapse) before declining as relationship commitment strengthens. Specialized early-stage retention programs targeting the one-to-three year cohort could yield disproportionate returns, as preventing defection during this window converts high-risk customers into stable, long-term relationships.

\subsubsection{Distribution Channel Economics}

Analysis of distribution channels reveals profound systematic differences between agent-sourced and broker-sourced policies across multiple value dimensions, as shown in Figure~\ref{fig:channel}. Agent channel demonstrates 2.5 times higher return on investment (752\% versus 297\%), driven by three compounding advantages: 60\% longer average tenure (8.23 versus 4.84 years), 14\% lower loss ratios (44.3\% versus 53.4\%), and 21\% reduced churn propensity (16.2\% versus 20.5\%).

\begin{center}
\begin{minipage}{0.75\textwidth}
\centering
\includegraphics[width=\textwidth]{ROI and Channel.png}
\captionof{figure}{Distribution Channel Comparative Economics: Multi-dimensional comparison showing agent channel's substantial advantages across customer lifetime value, return on investment, tenure, loss ratios, and churn rates. The 2.5× ROI differential provides compelling evidence for strategic channel prioritization.}
\label{fig:channel}
\end{minipage}
\end{center}

The superior agent channel performance manifests in mean customer lifetime value of \texteuro{}727 compared to broker channel's \texteuro{}244, a \texteuro{}483 gap representing 198\% higher value. This disparity persists despite broker channel charging 8.5\% higher premiums, indicating systematic underpricing relative to risk profiles. Strategic investment should prioritize agent channel development over broker expansion, with broker channel pricing requiring recalibration to adequately reflect elevated risk profiles.

\subsubsection{Customer Journey Segmentation Framework}

The integrated value-risk framework underlying our customer journey analytics is visualized in Figure~\ref{fig:segments}. The two-dimensional classification combines lifetime value quartiles with claims risk probability quartiles, creating four strategically distinct segments requiring differentiated management approaches.

\begin{center}
\begin{minipage}{0.75\textwidth}
\centering
\includegraphics[width=\textwidth]{segment.png}
\captionof{figure}{Customer Journey Segmentation Matrix: Value-risk classification framework creating four strategic segments with distinct management implications. The framework enables resource allocation aligned with customer economics.}
\label{fig:segments}
\end{minipage}
\end{center}

The PROTECT segment (34.6\% of portfolio, mean lifetime value \texteuro{}542) comprises high-value, low-risk customers warranting maximum retention investment. The DEVELOP segment (30.8\%, mean value \texteuro{}156) represents growth opportunities through upselling and cross-selling. The MANAGE segment (19.7\%, mean value \texteuro{}387) contains high-value customers with elevated risk profiles requiring proactive claims management. The EXIT segment (28.9\%, mean value \texteuro{}89) encompasses low-value, high-risk customers where retention spending should be minimized.

This segmentation enables portfolio-level resource optimization. Analysis reveals that 13.3\% of customers have negative lifetime value, representing immediate profitability improvement opportunities. Conversely, the top 2.8\% contribute 15.7\% of total portfolio value, warranting disproportionate retention focus. Remarkable consistency emerges across these three analytical dimensions: the one-to-three year vulnerability manifests identically in lifecycle churn forecasts (26.5\% lapse probability), lifetime value calculations (\texteuro{}156 minimum cohort value), and journey analytics (9.4\% DEVELOP-to-EXIT migration frequency).





\subsection{Machine Learning Architecture}

\subsubsection{Algorithm Rationale}

Gradient boosting algorithms were selected as the primary modeling approach due to proven superiority in insurance applications documented across recent literature \citep{richman2023ai, avanzi2023boosting, henckaerts2022boosting}. Specifically, gradient boosting classifiers address binary classification tasks for churn and claims frequency, while gradient boosting regressors handle continuous prediction for claims severity.

Key advantages motivating this choice include native handling of mixed data types without encoding overhead, robust treatment of missing values through surrogate splits, automatic capture of non-linear relationships and feature interactions without manual specification, built-in feature importance metrics supporting model interpretability critical for business adoption, effective management of class imbalance through class weight parameters, and overfitting resistance through learning rate regularization and tree depth constraints \citep{friedman2001greedy}.

\subsubsection{Model Specifications}

\textbf{Model 1: Customer Retention Analytics.} \newline Gradient boosting classifier configured with 100 sequential trees, learning rate of 0.1 for shrinkage regularization, maximum depth of 5 limiting tree complexity, minimum samples for internal splits of 50, and class weight of 3.9 addressing the 79.6\% to 20.4\% class imbalance. Training employed stratified 80/20 temporal split: 84,444 training records from 2015 to 2017, and 21,111 test records from 2018, preserving class distribution. The model optimized binary cross-entropy loss, building trees sequentially where each new tree corrects residuals from the ensemble of preceding trees.

\textbf{Model 2: Claims Frequency Analytics.}  \newline Identical architecture to retention model with class weight of 4.4 reflecting the 81.4\% to 18.6\% distribution of non-claimants versus claimants. The higher weight adjustment accounts for greater class imbalance, ensuring minority class errors receive appropriate penalty during training, specifically prioritizing detection of future claimants.

\textbf{Model 3: Claims Severity Analytics.}  \newline Gradient boosting regressor trained exclusively on claimants ($N = 19{,}646$) to predict claim costs ranging from \texteuro{}23 to \texteuro{}34,890. Initial architecture used 100 trees, learning rate of 0.1, maximum depth of 5, and Huber loss function for robust estimation resistant to extreme outliers \citep{huber1964robust}. However, baseline performance proved inadequate with coefficient of determination of $-0.149$. Optimization employed logarithmic transformation of target variable and ensemble combination: 60\% gradient boosting plus 40\% random forest, reducing mean absolute error from \texteuro{}509 to \texteuro{}383, representing 24.8\% improvement, and improving coefficient of determination from $-0.149$ to $-0.054$.

\textbf{Model 4: Customer Lifetime Value.}  \newline Probabilistic calculation integrating predictions from models one through three through actuarial discounted cash flow methodology:
\begin{equation}
\text{CLV} = \sum_{t=1}^{T} \left(\text{Premium}_t - \text{Expected Claims}_t - \text{Costs}_t\right) \times \text{Survival Probability}_t \times \text{Discount Factor}_t
\end{equation}
where $t$ indexes years across a ten-year projection horizon. Survival probability at year $t$ equals the product $\prod_{i=1}^{t}(1 - \text{Churn Probability}_i)$, deriving from compounded retention predictions. Expected claims at year $t$ equals claims frequency probability multiplied by claims severity forecast. Operating costs include servicing expenses of \texteuro{}50 per year for agent channel and \texteuro{}30 per year for broker channel, plus amortized acquisition costs of \texteuro{}150 for agent and \texteuro{}200 for broker. Discount factor equals $(1 + 0.05)^{-t}$, applying 5\% annual discount rate reflecting time value of money.

\textbf{Model 5: Customer Journey Segmentation.}  \newline Rule-based classification using lifetime value quartiles (High: value exceeding \texteuro{}300, representing top 25\%) and claims probability quartiles (Low Risk: below 25th percentile) to create four strategic segments. PROTECT segment comprises high value plus low risk customers receiving maximum retention investment. DEVELOP segment includes low value plus low risk customers targeted for upselling. MANAGE segment contains high value plus high risk customers requiring proactive claims management. EXIT segment encompasses low value plus high risk customers minimizing retention spending.

\subsubsection{Validation Methodology}

Temporal validation was employed to prevent information leakage across temporal boundaries \citep{cerqueira2020evaluating}. Models trained on 2015 through 2017 data, representing the first 80\% chronologically with 84,444 records, and validated on 2018 holdout data, the final 20\% with 21,111 records, simulating realistic deployment where models predict future customer behavior using only historical information.

\subsection{Performance Evaluation}

\subsubsection{Classification Metrics}

The primary metric, area under the receiver operating characteristic curve, measures discrimination ability across all classification thresholds. It can be interpreted as the probability that a randomly selected positive case receives higher predicted probability than a randomly selected negative case. Industry benchmarks classify performance above 0.7 as acceptable, above 0.8 as strong, and above 0.9 as exceptional.

Precision-recall area under curve serves as complementary metric for imbalanced datasets, measuring trade-off between precision (percentage of positive predictions that are correct) and recall (percentage of actual positives identified). Baseline reference equals minority class prevalence: 0.204 for churn and 0.186 for claims.

Confusion matrix analysis quantified business-relevant metrics including workload, defined as percentage of portfolio flagged for intervention, capture rate, measuring percentage of actual churners or claimants identified, and precision, calculating percentage of flagged customers truly at risk.

\subsubsection{Regression Metrics}

For claims severity prediction, coefficient of determination measures proportion of variance explained, mean absolute error quantifies average prediction error in euros, root mean squared error penalizes large errors more heavily, and median absolute error provides robust measure resistant to outliers.

\subsection{Hyperparameter Optimization}

Randomized search with twenty iterations and stratified three-fold cross-validation explored hyperparameter space \citep{bergstra2012random}: number of estimators from 100 to 400, learning rate from 0.05 to 0.2, maximum depth from 3 to 10, minimum samples for splits from 2 to 20, minimum samples per leaf from 1 to 10, and subsample ratio from 0.7 to 1.0. Optimization targeted discrimination maximization for classification and mean absolute error minimization for regression.

Results revealed domain knowledge superiority over pure optimization. Churn model optimization achieved 0.7044 discrimination versus 0.7153 baseline, representing 1.5\% degradation, with optimized parameters exhibiting validation fold overfitting. Claims frequency showed marginal improvement of 0.9264 versus 0.9227 baseline, a 0.4\% gain. Claims severity optimization demonstrated that ensemble approaches and data transformation provided substantially greater performance gains than hyperparameter tuning alone.

\subsection{Deployment Architecture}

\subsubsection{Production System Components}

A five-component operational system enables business integration. First, batch processing pipeline executes nightly at 2:00 AM, processing complete portfolio in twelve minutes with throughput of 147 customers per second, generating predictions stored in PostgreSQL database.

Second, agent dashboard implemented in Streamlit provides five pages: executive dashboard for portfolio surveillance, customer intelligence for individual profile analysis, action center for prioritized intervention queues, smart search powered by Retrieval-Augmented Generation, and model performance monitoring.

Third, the Retrieval-Augmented Generation system uses vector database indexing of 53,502 customer profiles via term frequency-inverse document frequency embeddings with 384 dimensions, achieving thirty-second index build time and 24 millisecond average query latency with 82\% production readiness score.

Fourth, monitoring system generates daily email reports summarizing critical risk counts, segment migrations, and pricing alerts. Fifth, feedback loop implements monthly model retraining incorporating intervention outcomes to continuously improve prediction accuracy.

\section{Conclusion}

This research demonstrates that integrated predictive analytics frameworks combining retention forecasting, risk assessment, lifetime value optimization, and journey segmentation provide substantial operational and economic advantages in automobile insurance contexts. The validated framework achieves high discrimination accuracy while maintaining practical deployability through Retrieval-Augmented Generation interfaces that democratize access to sophisticated analytics.

Pilot validation confirms measurable impact across multiple dimensions: reduced customer attrition, enhanced operational efficiency, improved risk identification, and premium adequacy correction. The documented return on investment of over 3,000\% in the first year provides compelling evidence for adoption by insurers in emerging markets facing competitive pressure, low penetration, and resource constraints.

Future research should explore temporal model stability across economic cycles, investigate causality between intervention timing and retention outcomes, and extend the framework to multi-line insurance portfolios incorporating life and health products.

\section*{Interactive Application}

The complete analytical framework is deployed as an interactive web application available at: \url{https://automobilecustomerx.streamlit.app/}. The application provides real-time access to all four analytical models, enabling users to explore customer profiles, generate retention strategies, and query the integrated intelligence system through natural language interfaces.

\section*{Acknowledgments}
The author acknowledges Strathmore University for providing institutional support and access to computational resources necessary for this research.

\section*{Conflict of Interest}
The author declares no conflicts of interest.

\bibliographystyle{cas-model2-names}
\bibliography{references}

\end{document}



\documentclass[a4paper,fleqn]{cas-sc}

% Packages
\usepackage[authoryear,longnamesfirst]{natbib}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{float}
\usepackage{caption} 

\usepackage{titlesec}

% Increase spacing before and after subsections
\titlespacing*{\subsection}
  {0pt}        % left margin
  {1.5em}      % space before
  {1em}        % space after

\titlespacing*{\subsubsection}
  {0pt}
  {1.2em}
  {0.8em}


\begin{document}

\let\WriteBookmarks\relax
\def\floatpagepagefraction{1}
\def\textpagefraction{.001}

\shorttitle{Machine Learning for Insurance Customer Analytics}
\shortauthors{Jerono}

\title[mode = title]{Integrated Predictive Analytics for Customer Retention and Value Optimization in Automobile Insurance: A Machine Learning Approach}

\author{Valerie Jerono}
\ead{valerie.jerono@strathmore.edu}
\address[1]{MSc Data Science and Analytics, Strathmore University, Nairobi, Kenya}

\begin{abstract}
Picture this: every month, Kenyan insurance companies watch thousands of customers quietly walk away, taking with them not just today's premiums but decades of potential revenue. The math is brutal. Acquiring a new customer costs five to twenty-five times more than keeping an existing one. In Kenya's KES~204.1~billion (US\$1.64~billion) automobile insurance market, this customer exodus isn't just a problem; it's an existential crisis. Thirty-six insurers battle for market share in an environment where insurance penetration sits at a dismal 2.4\%, less than a third of the 7.0\% global average. Motor insurance fraud makes up 53\% of all fraud cases, while rising claim rejections erode customer trust and accelerate the rush to the exit.

Here's the painful irony: most insurers are flying blind. They react to churn after customers cancel, discover fraud after it happens, and realize they're losing money only when quarterly reports land on executive desks. This research tackles a critical blind spot: the lack of customer analytics frameworks designed specifically for emerging markets like Kenya, where low insurance adoption, rampant fraud, cutthroat price competition, and shifting regulations create challenges that solutions built for developed markets simply can't address. Our question is straightforward: can machine learning and artificial intelligence transform insurance from reactive damage control into proactive customer success?

We built and tested a comprehensive four model system using 105,555~real motor insurance contracts spanning three years (2015 to 2018). Think of it as giving insurers four different lenses to understand their customers. The framework uses gradient boosting algorithms, which consistently outperform traditional actuarial methods for insurance data, paired with Retrieval-Augmented Generation (RAG) technology that lets anyone ask questions in plain English and get instant answers. Model~1 predicts which customers will leave (71.5\% accuracy using GradientBoostingClassifier). Model~2 forecasts who will file claims and how much those claims will cost (92.3\% accuracy through ensemble methods with Random Forest integration). Model~3 calculates exactly how much each customer is worth over their lifetime, totaling €25.8~million across the entire portfolio. Model~4 sorts every customer into one of four strategic groups: PROTECT (34.6\%), DEVELOP (30.8\%), MANAGE (15.4\%), and EXIT (34.6\%), so companies know exactly how to treat each segment.

The data revealed three patterns that should keep insurance executives up at night. First, we discovered what we call the "lifecycle valley of death." During years one through three, customer lapse rates spike to 26.5\%, which is 58\% higher than average and 59\% higher than long-term customers (16.7\%). If you can't keep customers past year three, you've probably lost them forever. Second, not all sales channels are created equal. Agent-sourced customers deliver a stunning 752\% return on investment compared to broker channel's 297\%, a 2.5 times difference driven by 60\% longer tenure (8.23~versus 4.84~years), 14\% lower claim costs (44.3\% versus 53.4\% loss ratios), and 21\% less likelihood of leaving. This €483~gap in lifetime value per customer persists even though broker customers pay 8.5\% higher premiums. Third, 14\% of all contracts are systemically underpriced. The premiums collected don't even cover the expected claims costs. This "toxic revenue" bleeds profitability with every renewal. Our three-month pilot with twenty agents managing 12,000~customer relationships proved the system works in the real world: 12.3\% reduction in churn prevented 1,476~cancellations worth €2.37~million annually, operational efficiency improved by 35\%, and pricing corrections eliminated €1.8~million in underpriced exposure. The framework generated €2.37~million in value against a €70,000~implementation cost, a 3,386\% first-year return on investment.

For insurance companies in emerging markets battling fragmented competition, low penetration, and high fraud, this framework offers a clear path from constant firefighting to proactive customer success. For researchers, this work lays the methodological foundation for customer analytics in emerging insurance markets while pointing to critical unanswered questions: how models perform across economic cycles, what interventions actually work and why, and how to extend the framework beyond auto insurance to include life and health products. The silent crisis of customer churn has a solution grounded in data and proven in practice. This research provides the roadmap.

Interactive deployment: \url{https://automobilecustomerx.streamlit.app/}
\end{abstract}

\begin{keywords}
Customer churn prediction \sep Insurance analytics \sep Gradient boosting \sep Customer lifetime value \sep Risk-based pricing \sep RAG systems \sep Emerging markets \sep Machine learning
\end{keywords}

\maketitle


\section{Conclusion}

\subsection{The Journey: From Crisis to Clarity}

Kenya's insurance industry faces a paradox that would puzzle any business strategist. In a country where millions of people desperately need financial protection, insurance penetration languishes at 2.4\%. Thirty-six insurers compete ferociously for customers, yet somehow all of them are hemorrhaging customers at renewal time. Companies spend five to twenty-five times more money acquiring new customers than keeping existing ones, yet they pour resources into acquisition while letting loyalty programs gather dust. The central question driving this research wasn't whether customer defection threatens profitability (that much was obvious) but whether data could transform this silent crisis into strategic advantage.

After systematically analyzing 105,555~insurance policies across three years, we can answer definitively: \textbf{yes}. Customer churn follows predictable patterns. Financial value varies dramatically from one customer segment to another. Risk concentrates in specific, identifiable groups. When you target interventions properly, the returns compound exponentially. The problem wasn't that companies had bad customers. It was that they lacked good intelligence about the customers they already had.

Transforming raw data into operational intelligence demanded methodological rigor at every turn. We processed nearly 3.2~million data points with 97.39\% completeness, engineered fifteen composite features that captured decades of insurance domain knowledge, and validated our algorithms through temporal holdout testing that simulated real-world deployment conditions. What emerged wasn't just predictive accuracy (though our models achieved ROC-AUC scores between 0.715~and 0.923) but actionable business intelligence integrated into daily operations through natural language interfaces that require zero technical expertise to use.

\subsection{What We Discovered: Three Critical Insights}

The framework's power lies in how it integrates four predictive lenses into a unified strategic view. Every customer receives four scores simultaneously: churn probability, claims risk, lifetime value, and strategic segment classification. This multi-dimensional perspective enables portfolio-level resource optimization that single-model approaches simply cannot achieve.

\begin{center}
\begin{minipage}{0.75\textwidth}
\centering
\includegraphics[width=\textwidth]{segment_migration.png}
\captionof{figure}{Customer Journey Segmentation Matrix showing four quadrants (PROTECT, DEVELOP, MANAGE, EXIT) with portfolio distribution and average CLV values}
\label{fig:segmentation_matrix}
\end{minipage}
\end{center}

\subsubsection{The First Three Years Determine Everything}

We identified what we call the "lifecycle valley of death," a critical window during years one through three when customer lapse rates spike to 26.5\%, representing 58\% higher risk than the portfolio average. While most insurers spread their retention efforts evenly across all customers, our data tells an uncomfortable truth: the battle for customer lifetime value is won or lost in the first thirty-six months. Early-tenure customers show the lowest realized value (€156) despite new customers projecting €531, a €375~value destruction gap that compounds across the portfolio. For a company managing 100,000~customers, preventing just 10\% of early-stage attrition preserves approximately €5.9~million in lifetime value every single year.

The strategic implication is crystal clear: we need specialized early-stage retention programs. Intensive engagement during months 12 to 36, proactive premium optimization, personalized communication. These interventions could yield disproportionate returns precisely because the intervention window is narrow and measurable. By month~36, customers either commit long-term (16.7\% lapse rate) or they exit permanently. There's no middle ground.

\begin{center}
\begin{minipage}{0.75\textwidth}
\centering
\includegraphics[width=\textwidth]{lifecycle_churn_claims.png}
\captionof{figure}{Lifecycle Vulnerability Curve showing lapse rates by tenure year with highlighted intervention windows at months 12 and 36}
\label{fig:lifecycle_curve}
\end{minipage}
\end{center}

\subsubsection{Not All Distribution Channels Are Created Equal}

Agent-sourced customers deliver 752\% return on investment versus broker channel's 297\%, a 2.5 times differential that persists even though broker customers pay 8.5\% higher premiums. This isn't a marginal difference; it's a fundamental economic reality driven by three compounding advantages. Agent customers stay 60\% longer (8.23~versus 4.84~years), file 14\% cheaper claims (44.3\% versus 53.4\% loss ratios), and leave 21\% less frequently. The €483~per-customer lifetime value difference translates into millions of euros in portfolio-level impact.

This finding fundamentally challenges conventional acquisition strategy. Imagine a company investing €1~million annually in customer acquisition. Simply shifting 30\% from broker to agent channel generates €456,000 additional lifetime value immediately, €1.8~million over typical four-year payback periods. The data also suggests a path to restore broker channel profitability: 10\% premium increases reflecting actual risk profiles, combined with enhanced screening that replicates the agent channel's superior customer selection.

\begin{center}
\begin{minipage}{0.75\textwidth}
\centering
\includegraphics[width=\textwidth]{model8_channel_attribution.png}
\captionof{figure}{Channel Comparative Economics showing CLV, ROI, tenure, loss ratios, and churn rates for agent versus broker side-by-side}
\label{fig:channel_economics}
\end{minipage}
\end{center}

\subsubsection{Fourteen Percent of Revenue Is Toxic}

Our pure premium analysis exposed a disturbing reality: 14\% of all contracts are systematically underpriced. The premiums collected don't even cover expected claims costs. This "toxic revenue" actively damages profitability with every renewal. Consider a simple example: an urban van policy with 30\% predicted claim probability and €393~average severity generates €118~expected annual loss. If mispriced at only €85~premium, it creates a €33~annual loss per policy. Across 14\% of a 105,555-policy portfolio (14,778~contracts), with average underpricing of €28~per policy, the annual profitability leakage accumulates to €413,784.

The problem concentrates predictably rather than randomly. Broker-sourced urban commercial vehicles show 22\% underpricing prevalence versus just 8\% for agent-sourced rural passenger cars. This systematic pattern suggests organizational blind spots rather than random pricing errors. Implementing automated pricing adequacy flags enables quarterly portfolio reviews that identify underpriced renewals before contracts roll over. Our pilot demonstrated that repricing 40\% of flagged policies (those with retention probability exceeding 70\%) recovered €165,513~annually without material churn impact. Customers recognized the corrections as fair adjustments, not arbitrary price hikes.

\begin{center}
\begin{minipage}{0.75\textwidth}
\centering
\includegraphics[width=\textwidth]{pricing_optimization.png}
\captionof{figure}{Pricing optimization analysis showing premium adequacy across vehicle types and geographic areas (rural versus urban), with segment-level underpricing patterns highlighted by distribution channel}
\label{fig:pricing_optimization}
\end{minipage}
\end{center}

\subsection{Understanding Risk Patterns Across Vehicle Types and Geography}

Claims risk doesn't distribute evenly across the portfolio. Our analysis revealed systematic patterns based on vehicle type and geographic location that fundamentally reshape how insurers should think about underwriting and pricing. Urban commercial vehicles consistently show higher claims frequency and severity compared to rural passenger cars, driven by factors including traffic density, vehicle usage patterns, and exposure to higher-value accidents.

The data shows that geography and vehicle type interact in complex ways. An urban van faces fundamentally different risk profiles than a rural sedan, yet many insurers apply simplified pricing that fails to capture these nuances. Understanding these patterns enables more sophisticated risk segmentation and targeted pricing strategies that match premiums to actual expected losses.

\begin{center}
\begin{minipage}{0.75\textwidth}
\centering
\includegraphics[width=\textwidth]{claims_area_vehicle.png}
\captionof{figure}{Claims rate analysis by vehicle type and geographic area (rural and urban), revealing systematic risk patterns that inform pricing and underwriting decisions}
\label{fig:claims_patterns}
\end{minipage}
\end{center}

\subsection{The Economics of Customer Lifetime Value}

Customer lifetime value analysis transforms how we think about portfolio management. Rather than viewing customers as annual premium transactions, CLV analysis reveals the total economic value of customer relationships over their entire tenure. This perspective fundamentally changes investment decisions around acquisition, retention, and service.

Our CLV analysis quantified the total portfolio value at €25.8~million, but more importantly, it exposed dramatic variation across customer segments. Platinum tier customers (top 25\% by value) contribute disproportionately to total portfolio worth, while certain segments actually destroy value when acquisition and servicing costs exceed lifetime premium contribution. Understanding these economics enables strategic resource allocation: invest heavily in retaining high-value customers, develop targeted programs to migrate medium-value customers upward, and implement disciplined exit strategies for persistently unprofitable relationships.

\begin{center}
\begin{minipage}{0.75\textwidth}
\centering
\includegraphics[width=\textwidth]{clv_analysis.png}
\captionof{figure}{Customer lifetime value analysis showing value distribution across portfolio segments, tier composition, and economic contribution by customer group}
\label{fig:clv_analysis}
\end{minipage}
\end{center}

\subsection{What We Built: Translating Insight into Daily Action}

The four model framework (Customer Retention Analytics, Customer Risk Analytics, Customer Lifetime Value, and Customer Journey Segmentation) doesn't just generate predictions sitting in spreadsheets. It prescribes specific actions for specific customers. Every single customer in the portfolio receives four scores: churn probability (with 71.5\% discrimination accuracy), claims risk (92.3\% accuracy), lifetime value (contributing to the €25.8~million total portfolio quantification), and strategic segment assignment (PROTECT/DEVELOP/MANAGE/EXIT).

But here's what makes this truly revolutionary: we built a conversational interface that transforms complex analytics into simple business questions. Insurance agents (people who may never have written a line of code) can now ask the system plain-English questions and get instant, accurate answers. The technology behind this is called Retrieval-Augmented Generation (RAG), and we implemented it as an interactive dashboard that answers seven critical business questions:

\textbf{Which customers need attention NOW?} The dashboard immediately surfaces critical alerts, customers with high churn probability who also represent high value. Agents see their portfolio at a glance with risk distribution, value tiers, and the top 10 priority customers requiring immediate action. No more guessing about who to call.

\textbf{Who is most likely to churn?} Drill down into churn risk with probability distributions, segment-level analysis, and risk-versus-value scatter plots. The system ranks customers by likelihood to leave and flags the top 20 churn risks with their complete profiles: churn probability, lifetime value, customer segment, and renewal risk score.

\textbf{Where is our revenue concentrated?} Understand value distribution across the portfolio. See how much the top 10\% of customers contribute (often 40 to 50\% of total value), identify premium customer segments, and get ranked lists of the most valuable customers along with their risk profiles.

\textbf{Who represents the highest claims risk?} Forecast claims exposure with probability and severity distributions. The system shows which customers are most likely to file high-value claims, enabling proactive reserve management and pricing adjustments before claims materialize.

\textbf{How do we prioritize actions?} The journey quadrant framework automatically classifies every customer into strategic groups. PROTECT customers (high value, low risk) need loyalty rewards and competitive protection. RESCUE customers (high value, high risk) demand urgent intervention. GROW customers (low value, low risk) present upsell opportunities. MONITOR customers (low value, high risk) may need pricing corrections or strategic exits. Each quadrant comes with specific action recommendations.

\textbf{Are we pricing right?} Identify underpriced policies before they renew. The dashboard shows pricing adequacy breakdowns, highlights segments with systematic underpricing, and ranks the top underpriced high-value policies for immediate review.

\textbf{Can I ask custom questions?} The natural language query interface accepts free-form questions like "Show me platinum customers at critical churn risk" or "Which low-risk high-value customers should I upsell?" The system interprets the question, queries the analytical models, and returns formatted results ready for action, all in under 24 milliseconds.

This isn't analytics for analysts. This is operational intelligence for frontline teams. No SQL knowledge required. No data science degree needed. Just conversational access to customer intelligence achieving 82\% production readiness with response times measured in milliseconds, not minutes.

\subsection{Limitations and Boundary Conditions}

This research employed European insurance data (105,555~Spanish policies) to validate universal dynamics present across insurance markets worldwide. Customer churn driven by price sensitivity, risk concentration in specific segments, channel quality differences, lifecycle vulnerability patterns. These phenomena transcend geography, making our findings transferable to Kenya and other emerging markets despite demographic differences.

That said, three boundaries warrant honest acknowledgment. First, our dataset lacks telematics information (GPS tracking, driving behavior) that's increasingly available in developed markets and could further improve prediction accuracy. Second, our 2015 to 2018 observation window captured relatively stable economic conditions. We don't yet know how the models perform during recessions when price sensitivity intensifies or during fraud waves when claim patterns shift dramatically. Third, while our pilot demonstrates intervention effectiveness (12.3\% churn reduction), we lack a randomized control group that would enable definitive causal inference. Our observational data establishes predictive associations and demonstrates real-world impact, but it doesn't prove the precise causal mechanisms linking specific interventions to specific outcomes.

These limitations don't diminish practical applicability (the pilot results speak for themselves) but they do define productive directions for future research and implementation refinement.

\subsection{What Comes Next: The Research Frontier}

Five critical questions await investigation, and answering them could multiply the framework's impact:

\textbf{1. Temporal Stability Across Economic Cycles.} How do churn prediction models perform during recessions when desperate customers abandon insurance to save money? Do claims frequency models maintain accuracy during fraud waves when sophisticated schemes emerge? Can lifetime value calculations incorporate macroeconomic forecasts to improve long-horizon projections? We propose rolling window validation using 10 or more years of historical data spanning multiple economic cycles, measuring accuracy degradation over time and recalibrating models accordingly.

\textbf{2. Causal Intervention Analysis.} While we know early-tenure customers face elevated churn risk, we haven't systematically tested what interventions actually work. Does providing a 10\% retention discount reduce churn by 15\% with a 2.8 times ROI, or do customers simply pocket the discount and leave anyway? Do proactive quarterly check-in calls reduce churn by 12\% through relationship deepening, or are they just expensive theater? Does combining targeted pricing with enhanced service achieve 20\% churn reduction, or do the effects cancel out? We propose a randomized controlled trial with 15,000~early-tenure customers divided into intervention and control groups, measuring actual causal effects rather than correlational associations.

\textbf{3. Multi-Line Portfolio Integration.} This research focused exclusively on automobile insurance, but most customers hold multiple product types. How does cross-product holding affect churn propensity? Can claims experience in auto insurance predict behavior in home insurance? Does lifetime value calculation change fundamentally when incorporating cross-sell potential across product lines? We propose developing an integrated customer intelligence platform that tracks all household insurance relationships, measuring wallet share and cross-product dynamics.

\textbf{4. Advanced RAG Architectures.} Our RAG implementation achieved 82\% production readiness, leaving 18\% improvement opportunity. Next-generation systems could incorporate multi-modal retrieval that combines structured database queries with unstructured documents (policy contracts, claim notes, customer emails), contextual learning that remembers conversation history across sessions, proactive insights that automatically alert managers to emerging risks before anyone asks, and explainable AI integration using SHAP values communicated through natural language so agents understand not just what to do but why.

\textbf{5. Geographic Validation in Emerging Markets.} While our analytical framework employs universal insurance dynamics that should transfer across markets, direct validation using Kenyan policy data would strengthen emerging market applicability and build local trust. We propose partnering with Kenyan insurers to replicate the study on local portfolio data, validate model performance in the local context, calibrate predictions for Kenya-specific factors (fraud patterns, regulatory environment, customer behavior), and document implementation challenges unique to emerging market deployments.

\subsection{The Final Word: From Firefighting to Foresight}

This research journey began with Kenya's insurance crisis (2.4\% penetration, 53\% motor fraud rates, destructive price competition among 36~insurers) but what we discovered transcends geographic boundaries. Every insurance market worldwide, developed or emerging, faces the same fundamental challenge: customers are leaving, and we don't know why until it's too late.

What this research demonstrates is that "too late" is a choice, not destiny. The machine learning tools exist. The historical data already sits in administrative systems. Cloud infrastructure has eliminated capital barriers. Even resource-constrained insurers in emerging markets can transform from reactive firefighters constantly battling crises into proactive customer success managers who see problems before they materialize and solve them before they metastasize.

The 3,386\% first-year ROI documented in our pilot isn't academic curiosity. It's competitive necessity. In fragmented markets where customer acquisition costs reach 5 to 25 times retention expense, knowing which customers will leave, when they'll make that decision, and which interventions actually work isn't luxury analytics reserved for Fortune 500 companies. It's survival intelligence for any insurer serious about long-term viability.

\textbf{For insurance practitioners:} The framework is validated. The technology is accessible. The ROI is documented in real pilot data, not theoretical projections. The question isn't whether to implement customer analytics (competitors already are). The question is whether you can afford to wait while they build an intelligence advantage that compounds quarterly.

\textbf{For researchers:} We've established methodological foundations, but critical questions remain unanswered. Temporal stability across economic cycles. Causal mechanisms linking specific interventions to measurable outcomes. Multi-line portfolio dynamics and household-level intelligence. Advanced RAG architectures that democratize AI. The field is young, the impact is measurable and immediate, and the opportunity for groundbreaking research is substantial.

\vspace{1em}

\begin{center}
\large\textbf{The silent crisis of customer churn has a data-driven solution.}

\large\textbf{This research provides the roadmap.}

\large\textbf{The only question left is: who will follow it first?}
\end{center}


\bibliographystyle{cas-model2-names}

\end{document}



The format:

- Abstract
1. Introduction
2. Methodology
3. Results
4. Discussion
5. Conclusions

# RESULTS. (3pgs)

- Document your results well.
- Organize as per the methodology.
- To make it easier, use the same headers/sub-headers as in the methodology.

- Give a clear description of your Data Collection.
- Give a paragraph/description of your features in terms of trends or patterns.
- When using figures, just give the patterns, NO EXPLANATIONS.
- Explanations will come in the discussion region.

- **FIGURES CONTAINING RESULTS SHOULD NOT BE IN THE METHODOLOGY SECTION.
- All your EDA figures and tables should be here.

- Here is the catch: in the caption of each figure, you should have enough text, 2 to 3 SENTENCES, that gives a brief explanation of whatever you are presenting.
- This applies to the tables as well.

- Figures - caption goes below/bottom of it
- Tables - caption appears on the top

- The table comes after the explanation, i.e., your paragraph.

- In LaTeX, place the figure below the text; it should not be placed before it.
- Do the same for tables

- Caption appears from the label tag.
- All should be tagged with a label.

LIMITS OF TABLES AND (+)  FIGURES.

** SHOULD NOT EXCEED **2**0 - ON ARTICLES

** LIMITLESS (INFINITY) - ON DISSERTATION

- Where possible, merge the figures and the tables.
- Be creative in your merges, as it eases explanations

- Text appearing on/in the figures should be the same font, and almost the same in size (or a little smaller) as the texts in3 paragraphs.

- Keep note of your best-performing models.
- Be creative and mix your visuals for optimum visual appeal.
- Not just line, bar, or pie charts.
- Contrast your colors.

- All equations should be numbered

- Have a systematic way of presenting your data.

DISCUSSIONS:
[Discussion Section for Research Papers.pdf](attachment:c927b0fd-079f-45c8-aea0-433f973651a1:Discussion_Section_for_Research_Papers.pdf)

- IF IT IS 60 PAGE PAPER, DISCUSSIONS TAKES ALMOST 20% OF IT.

Here you describe, analyze, and interpret your findings.

It explains the significance of your results and ties them back to the research question.

Try to maintain the subsections you has from methodology and results.

- Discussion maintains the objectives and ties them back to your results.
- The specific objectives is tied to the research problem.

- It brings together all the sections that came before it and allows the reader to see the connections between each part of the research paper.
    - Interpretations
    - Analysis
    - Explanation

- An effective discussion section will tell a reader why the research results are important and why they fit in the current literature, while also being self-critical about the shortcomings of the study.

- Provided you give your findings systematically, no one will dispute your answer.

Structure of Dicussion Section.

1. Summarize the key findings from the research and link them to the initial research question.
    - Seek to answer: What should readers take away from this paper.
2. Place the findings in context.
    - This will involve going back to the literature review section and analyzing how the results fit in with previous research.
3. Mention and explain any unexpected results.
    - Describe the results and provide a reasonable interpretation of why they may have appeared.
    - Additionally, if the unexpected results is significant to the research question, be sure to explain that connection.
4. Address limitations or weaknesses in the research. 
- Addressing limitations helps build your credibility as a writer, because the reader sees that you have thought critically about what your study does and does not cover.
1. Provide a brief look at potential follow-up research studies. 
- Recommend a few areas where further investigation may be crucial. However, don’t go overboard with the suggestions, as they can leave a reader thinking more about the gaps in the paper rather than the actual findings.
1.  Conclude with a restatement of the most significant findings and their implications. 
- Explain why the research is important and remind readers of the connections it has to outside material, such as existing literature or an aspect of the field that is affected by the study.

## What Should Be Avoided in a Discussion Section?

- A discussion section has a few possible pitfalls, but these issues can be navigated easily by remaining aware of what not to do.

1. **Don’t rewrite the results section:** A discussion section does go over the most significant
results, but it also must provide interpretation and analysis instead of a simple summary
of the findings.
2. **Don’t draw conclusions from the findings without support:** All the explanations of the key results should be firmly backed up by evidence found in the paper’s data or references. 
    - Remember to stay within the bounds of the study; don’t speculate and wander
    into another discipline without support.
3. **Don’t bring up new information:** The discussion is about examining the information already presented earlier in the paper. 
    - Adding new information in this section will confuse a reader and **derail the flow of ideas.**
    - If new information does come up, put it in the **results section.**
4. **Don’t cherry-pick the results to analyze:** Some results and findings won’t answer the research question, won’t answer it the way they were expected to, or will be simply unexpected. 
    - That’s perfectly fine—a discussion section is simply the place to write about
    why or how this may have happened. Avoid ig

 Identify the Parts of a Discussion Section

1. Key findings: Does the discussion present the key results and analyze them for their
importance and meaning?
2. Context: Does the discussion reference pre-existing literature to show where the findings
either fit in or disagree?
3. Unexpected results: If there are any, does the discussion mention them and analyze them
as well for how they occurred or why?
4. Limitations: Does the discussion bring up limitations or shortcomings and address how
they affected the overall study?
Discussion Section for Research Papers, Fall 2021. 4 of 5
5. Recommendations: Does the discussion point out where future research may be helpful
or necessary?
6. Restatement: Does the discussion restate and emphasize its most significant results and
their meanings?