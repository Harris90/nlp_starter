# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 18:17:17 2020

@author: prasa
"""

import nltk
from nltk.stem import  WordNetLemmatizer
from nltk.corpus import stopwords
import  re

paragraph ="""The registration of multiple police cases in Uttar Pradesh on charges of conspiracy and sedition is an insensitive,
albeit unsurprising, response to the legitimate outrage sparked by the way in which its officials had handled 
the aftermath of the gang rape and murder of a Dalit girl in Hathras. The State administration is seeking to 
convrt the outcry and political advocacy into a putative conspiracy to foment caste discord. There seems to
 be inadequate recognition that initial attempts to deny that any rape took place and to prevent political 
 leaders and the media from meeting the girl’s family pointed to an administrative posture hostile to the 
 doing of complete justice. It was only to be expected that such an attitude would create a backlash against 
 the State government. The only way to restore its image is to display empathy for the victim and treat protests
 seeking justice as legitimate. Even though the suspects named by the 19-year-old before her death in a Delhi 
 hospital have been arrested and the FIR includes the charge of gang rape, there are clear signs that the 
 Yogi Adityanath government is adopting dubious tactics to prevent what it sees as the crystallisation of 
 public opinion against itself. Recognising that the formation of an SIT did not help undo the damage to 
 the government’s image, the Chief Minister recommended a CBI investigation. However, even that came across 
 as a move to fob off any adverse order from the Allahabad High Court, which has taken suo motu cognisance 
 of the matter."""
 
 
sentences_lemt = nltk.sent_tokenize(paragraph)
word_lemmatizer = WordNetLemmatizer()
corpus=[]

for i in range(len(sentences_lemt)):
    review = re.sub('[^a-zA-Z]', ' ', sentences_lemt[i])
    review_lower = review.lower()
    review_split = review_lower.split()
    review_split = [word_lemmatizer.lemmatize(word) for word in review_split if not word in set(stopwords.words('english'))]
    review_split = ' '.join(review_split)
    corpus.append(review_split)

# now converting words to vector using tfidf

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer()
X_tfidf= tfidf_vect.fit_transform(corpus).toarray()
