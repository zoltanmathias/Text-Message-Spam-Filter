# Text-Message-Spam-Filter
Identify spam text messages with the use of Naive Bayes. 

Mobile phone spam is limited to 160 characters and is subject to abbreviations, making it harder to detect than email spam, where the bayesian algorithm has been successfully applied.

Data set used: http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/

The data is split into two columns: '1' for identification (ham or spam), and '2' for the actual text message.
There are 5,574 samples of which 86.6% are ham and 13.4% are spam.

The python code scraps common words that add no value to prediction such as 'are', and 'the'.
The remaining words are vectorized and given values based on their frequency.
This 'bag of words' is then used to detect whether a certain message is spam.
