# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:27:20 2020

@author: prasa
"""

import nltk
from nltk.stem import  PorterStemmer
from nltk.corpus import stopwords

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
 
 
sentences = nltk.sent_tokenize(paragraph)
stemmer = PorterStemmer()


for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)