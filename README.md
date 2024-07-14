# HATE-SALVINI
AP project work for Computational Modeling (WiSe 2023/2024). Project description:

Detecting hate speech on social media has become crucial over the last few years. Platforms like Twitter are often used by political figures to expand their political discourse to a wider audience. Their communication, however, might sometimes convey populist and xenophobic stances. The present research aims to analyze Italian politician Matteo Salvini’s tweets to investigate, with the help of Natural Language Processing (NLP) techniques, how he discusses and frames immigrants and immigration as a phenomenon, and whether nuances of xenophobia can be found in his tweets. Results showed that immigration-related issues are among the most discussed topics in Salvini’s Twitter communication, and are generally framed negatively and as a concern to society. Sentiment and Emotion classification revealed a predominantly negative tone with anger as the prevalent emotion whenever he discusses immigration-related matters. These results are in line with previous qualitative studies and highlight Salvini’s use of populist and anti-immigration rhetoric.

# Repository Structure

- `/data`: the data used for this project, including the uncleaned original data and the cleaned version

- `/figures_and_plots`: contains the figures, ploths and graphs obrained with the computations and contained in the report

- `/notebooks`: contains a notebook with the whole workflow

- `/report`: project report and LaTeX code

- `/requirements`: contains the requirements.txt file

- `/scripts`: contains the workflow but in scripts

# How To Run

Install runtime requirements in `/requirements`.
Run either **entire_workflow.ipyn** in /notebooks or the scripts in `/scripts` (extract_tweets.py > topic_modleing.py > frequency_distribution.py > sentiment_and_emotion_classification.py)

# License

All source code is licensed under the `MIT License`.
