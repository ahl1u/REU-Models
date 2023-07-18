# REU-Models
some NLP models I used during my summer research

Code for training roBERTa models - roBERTa_training_sentiment.py, roBERTa_training_user.py

Code for applying trained roBERTa models - sentiment_roBERTa.py, user_roBERTa.py

Some code that I tested prior to settling on roBERTa - others files that start with 'sentiment_'

There is also some code for calculating cohen kappa score, this was used to determine agreement rate when labeling tweets - kappa_score.py

Code for twitter profile pic categorizer, wherin I slightly modified the existing script - faceapp_demog.py

Lastly, code for topic modeling, LDA for determining optimal number of topics regarding clarity and overlap with coherence score graphs and intertopic distance maps with BERTopic for actual topic modeling - LDA.py, BERTopic.py