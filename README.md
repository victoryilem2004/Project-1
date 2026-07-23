# Project-1
# Where Are the ML Jobs, Really? A Data-Driven Look at US Postings

## Overview

This project analyzes 819 US-based machine learning job postings (1000 in the raw dataset, before filtering out non-US locations and invalid
entries), sourced from Kaggle. It started as an attempt to find the "best hiring season" for ML roles. Early analysis revealed that goal wasn't answerable with this data, so the project shifted toward questions the data could actually support with confidence: where ML jobs are concentrated, when during the week they're posted, what industries are hiring, and what natural groupings exist in job titles and company
descriptions.

The full step-by-step analysis lives in `eda.ipynb`. This README summarizes the findings and the reasoning behind them.

## A key early finding: the data can't answer the original question

The dataset spans December 2022 through April 2025, but that range is extremely misleading at first glance. Breaking postings down by year shows:

| Year | Postings |
|------|----------|
| 2022 | 1 |
| 2023 | 2 |
| 2024 | 74 |
| 2025 | 920 |

Within 2025, postings are almost entirely concentrated in March and April (425 and 433 postings respectively), with every other month in single or low double digits. This is very likely a snapshot bias: job boards mostly show currently active postings, so a scrape naturally captures far more of whatever was recently posted than anything from further back, since older postings get taken down once filled. This is not evidence that March and April are genuinely the busiest hiring months, it's most likely an artifact of when the data was collected.
Because of this, month-of-year is not a reliable signal in this dataset, and the "best hiring season" question is left unanswered, honestly, rather than forced into a misleading conclusion.

## Finding 1: Geographic concentration

California accounts for the overwhelming majority of postings, roughly 9 times more than the second-place state (New York). At the city level, 5 of the top 10 cities (San Francisco, Menlo Park, San Jose, Mountain View, Los Gatos) are all in the San Francisco Bay Area specifically, showing the concentration is tighter than "California" alone suggests, it's really a Bay Area story.

## Finding 2: Day-of-week timing (a more reliable replacement for "season")

Since month-of-year isn't trustworthy, day-of-week was used instead as a timing signal, it doesn't depend on having even coverage across the year, only across weekdays, which the dataset does have. Wednesday stands out with a clear spike (222 postings, notably above neighboring
days). Note that since roughly 92% of the dataset falls within March-April 2025, this pattern primarily reflects that window rather than year-round behavior. Whether the Wednesday spike reflects genuine hiring behavior or scraping timing can't be confirmed with certainty from this data alone.

## Finding 3: Industry patterns (two methods, compared)

There's no industry column in the raw data, so industry was inferred two different ways, and then compared.

**Method 1: Manual classification.** A rule-based approach combining known-company lookups (for high-volume employers like TikTok, Meta, and Amazon, where the industry is common knowledge) with keyword matching against company description text for everything else. This left about 14% of postings unclassified, mostly companies with vague, brand-driven marketing copy rather than descriptive text.

**Method 2: Unsupervised clustering.** TF-IDF plus K-Means clustering on company descriptions, with no predefined categories or company knowledge given to the algorithm. Clustering directly on posting-level data mostly rediscovered exact duplicate text from high-volume posters (Meta, TikTok, Netflix each formed their own cluster simply because their description repeats identically across postings). Deduplicating to one row per company revealed clearer, more genuine structure: coherent clusters emerged for IT staffing/recruiting, healthcare/medtech, and AI/ML product companies.

**Comparing the two**, a cross-tab between manual labels and cluster assignments showed strong, independent agreement for IT Staffing/Consulting and Healthcare, both methods converged on these categories without referencing each other. Agreement was weaker for AI/ML specifically, clusters that looked like "AI/ML products" from manual inspection turned out to scatter across AI/ML, Enterprise Software, and
Other/Unmatched labels once checked numerically, suggesting the algorithm may be partly separating companies by product type (platform vs. tool) rather than industry alone.

## Finding 4: Job title clustering (unsupervised)

Job titles were converted into TF-IDF vectors and grouped into 8 clusters using K-Means, with no predefined categories. The algorithm independently separated titles along several real dimensions: seniority (Senior ML Engineer as its own cluster), career stage (internships/entry-level separated from experienced roles), specialization (AI/Generative AI-focused roles vs. general ML Engineer roles), and role type (Data Scientist vs. ML Engineer, with a distinct hybrid Data Science/Analytics cluster). PCA was used to visualize the clusters in 2D and 3D.

## Limitations

- The "best hiring season" question could not be answered due to a scraping-related recency bias in the date data.
- Industry classification depends on company description text quality; roughly 14% of postings remain unclassified after both methods.
- K-Means requires the number of clusters to be chosen manually (8 was used throughout); a different number might reveal coarser or finer
  groupings.
- Clusters reflect textual patterns, not verified real-world role or industry differences, some distinct-looking clusters may represent the
  same underlying reality described differently.
- The dataset is a single snapshot in time, not a representative sample across the full time range it nominally covers.

## Tools used

Python (pandas, matplotlib, scikit-learn) for cleaning, analysis, and modeling, in a Jupyter notebook via VS Code.

## Repository contents

- `1000_ml_jobs_us.csv` — original raw dataset
- `cleaned_ml_jobs_us.csv` — cleaned, US-only dataset with all derived
  columns (region, year, month, day of week, industry classification,
  title clusters)
- `eda.ipynb` — full analysis notebook
- `README.md` — this file
