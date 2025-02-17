Lesson 4
========

*   New and Exciting Content
    
*   Why Hugging Face transformer
    
    *   Will we in this lecture fine-tune a pretrained NLP model with HF rather than fastai library?
        
    *   Why use transformer rather than fastai library?
        
    *   Is Jeremy in the process of integrating transformer into fastai library?
        
    *   Does transformer has the same layered architecture of fastai? Is it high level enough?
        
    *   Why it is a good thing to use a reasonably high level library (not as high as fastai)?
        
*   Understand Fine-tuning
    
    *   Do we have the foundations to understand the details of fine-tuning now?
        
    *   How to understand pretrained model in terms of parameters confidence? 03:51
        
    *   Is fine-tuning trying to increase on the parameters which are not confident?
        
*   ULMFiT: the first fine-tuned NLP model
    
    *   Where this model was first developed and taught?
        
    *   Who wrote the paper?
        
    *   What’s its impact?
        
*   ULMFiT step 1: a language model from scratch
    
    *   What is the first language model in step one?
        
    *   What’s the model trying to predict? What’s the dataset?
        
    *   Why is this task so difficult? 06:10
        
    *   How much knowledge does the model have to understand in order to predict?
        
    *   How well can this first model predict in step one?
        
*   Step 2: fine-tuned the first model on IMDB
    
    *   How did Jeremy build the second language model?
        
    *   Where did the second model start with? What was the dataset for the second model?
        
    *   What was the second model good at predicting?
        
*   Step 3: turn a language model to classify
    
*   Labels of language models
    
    *   What are the labels for the datasets of the first two models?
        
*   Transformer models vs ULMFiT
    
    *   When did the transformers first appear?
        
    *   What’s transformers models are built to take advantage of?
        
    *   What is not transformers trying to predict? (reason in part 2)
        
    *   How transformers modified its dataset and what does it predict? 09:41
        
    *   Does ULMFiT and Transformers really differ much on what to predict?
        
    *   How much different are the 3 steps between ULMFiT and Transformers?
        
*   What a model knows
    
    *   What can lower and higher layers of parameters/weights learn? 11:08
        
    *   What we do to those layers of weights for transfer learning? 13:20
        
    *   Zeiler and Fergus paper
        
*   NLP beginner on Kaggle competition
    
    *   Using a Kaggle competition to introduce NLP for beginners, isn’t it amazing!
        
    *   Why we should take Kaggle competition more seriously? 15:06
        
    *   What real world tasks can NLP classification do? 15:57
        
*   Examine the competition dataset
    
    *   What is inside the competition dataset?
        
    *   How classificationish does the dataset look like?
        
    *   What do we predict about ‘anchor’ and ‘target’?
        
    *   What value to predict?
        
    *   Why it is not really a straightforward classification?
        
    *   What is the use of ‘context’?
        
*   Model Strategy
    
    *   How to modify the dataset in order to turn a similarity problem into a classification problem?
        
    *   Should we always try to solve a problem by turning it into a problem we are familiar with?
        
*   Get notebook ready
    
    *   When and how to use a GPU on Kaggle?
        
    *   Why Jeremy recommend Paperspace over Kaggle as your workstation?
        
    *   How easy has Jeremy made it to download Kaggle dataset and work on Paperspace or locally?
        
    *   How to do both python and bash in the same jupyter cell?
        
*   Get raw dataset into documents
    
    *   How to check what inside the dataset folder?
        
    *   Why it is important to read Competition data introduction which is often overlooked?
        
    *   How to read a csv file with pandas? 24:30
        
    *   What are the key four libraries for data science in python? 24:46
        
    *   What is the other book besides fastbook recommended by Jeremy? 25:36
        
    *   Why you must read it too?
        
    *   How to access and show the dataset in dataframe? 26:39
        
    *   How to describe the dataset? What does it tell us in general? 27:10
        
    *   What did the number of unique data samples mean to Jeremy at first? 27:57
        
    *   How to create a single string based on the model strategy? 28:26
        
    *   How to refer to a column of a dataframe in reading and writing a column data?
        
*   Tokenization: Intro
    
    *   How to turn strings/documents into numbers for neuralnet?
        
    *   Do we split the string into words first?
        
    *   What’s the problem with the Chinese language on words?
        
    *   What are vocabularies compared with splitted words?
        
    *   What to do with the vocabulary?
        
    *   Why we want the vocabulary to be concise not too big?
        
    *   What nowadays people prefer rather than words to be included in vocab?
        
*   Subwords tokenization by Transformer
    
    *   How to turn our dataframe into Hugging Face Dataset?
        
    *   What does HF Dataset look like?
        
    *   What is tokenization? What does it do?
        
    *   Why should we choose a pretrained model before tokenization?
        
    *   Why must we use the model’s vocab instead of making our own?
        
    *   How similar is HF model hub to TIMM? 33:10
        
    *   What Jeremy’s advice on how to use HF model hub?
        
    *   Are there some models generally good for most of practical problems? 34:17
        
    *   When did NLP models start to be actually very useful? 34:35
        
    *   Why we don’t know much about those models which potentially are good for most of things?
        
    *   Why should we choose a small model to start with?
        
    *   How to get the tokens, vocabs and related info of the pretrained model? 36:04
        
    *   How to tokenize a sentence by the model’s style?
        
    *   After a document is splitted into a list of vocab, do we turn the list of vocab into a list of numbers? Numericalization 38:30
        
    *   Can you get a sense of what subword vs word is from the examples of tokenization
        
    *   How to tokenize all the documents with parallel computing? 38:50
        
    *   Given the input column is the document, what’s inside the input\_id column?
        
*   Special treatment to build input?
    
    *   Do we need to follow some special treatment when building a document or an input from dataset?
        
    *   What about when the document is very long?
        
*   Start ULMFiT on large documents
    
    *   What ULMFiT is best at doing?
        
    *   Why ULMFiT can work on large documents fast and without that much GPU?
        
    *   How large is large for a document?
        
*   Some obscure documentations of Transformer library
    
*   The most important idea in ML
    
    *   Is it the idea of having separate training, testing, validation datasets?
        
*   Underfitting vs Overfitting
    
    *   How to create a function to plot a polynomial function with a degree variable?
        
    *   What are 1st/2nd/3rd degree polynomial?
        
    *   What does Jeremy think of sklearn? When to use it? 47:37
        
    *   What is underfitting? Why a too-simple model is a problem or is systematically biased? 48:12
        
    *   What is overfitting? What does an overfit look like? 48:58
        
    *   What is the cause of overfitting?
        
    *   It is easy to spot underfitting, but how to filter an overfitting from the function we want?
        
*   Validation: avoid overfitting on training set
    
    *   How to get a validation dataset and use it?
        
    *   Why you need to be careful when use other libraries other than fastai?
        
*   How and Why to create a good validation set
    
    *   Did you know simply random 20% of dataset as a validation set is not good enough?
        
    *   For example, shouldn’t you select validation dataset so that your model can predict the future rather than the past?
        
    *   Why is Kaggle competition a great and real-world way to appreciate using validation set to avoid overfitting?
        
    *   How validation set can help avoid overfitting in 2 Kaggle competition on real world problems? 54:44
        
    *   Watch out when touching cross-validation 56:03
        
    *   Why should you be careful when simply using library-ready tools of selecting validation set randomly?
        
    *   Validation post by Rachel
        
*   Test set: avoid overfitting on validation set
    
    *   What is a test set for?
        
    *   Why need it when we have a validation set?
        
    *   When or how can you overfit on a validation set? or
        
    *   Why is validation set not enough to overcome model overfitting?
        
    *   Why Kaggle prepares two test sets? or
        
    *   Why Kaggle thinks that two test sets are enough to filter overfitting models
        
*   Metrics functions vs Loss functions
    
    *   How we use validation set to check on the performance of model?
        
    *   Will Kaggle competition choose the metrics for you?
        
    *   Should the metrics be our loss function?
        
    *   What kind of functions you should use as loss function? (bumpy vs smooth)
        
    *   So, always be aware: the loss your model tries to beat may not be the same function to rate your model
        
    *   Why one metric is always not enough and can cause much problem?
        
*   Metrics: you can’t feel it from math
    
    *   What is Pearson correlation (r) and how to interpret it?
        
    *   Which can teach you how r behave, its math function or its performance on datasets?
        
    *   Should we a plot with a 1000 random data point or a plot with the entire a million data points?
        
    *   How to get correlation coefficient for every variable to every other variable? 1:06:27
        
    *   How to read the correlation coefficient matrix?
        
    *   How to get a single correlation coefficient between two things?
        
    *   How to tell how good is a correlation coefficient number? 1:07:45
        
    *   What are the things to spot? (tendency line, variation around the line, outliers)
        
    *   How to create transparency on the plot?
        
    *   How can we tell from another example that r is very sensitive to outliers? 1:09:47
        
    *   How much can removing or mess up a few outliers really affect your scores on r? or
        
    *   Why do you have to be careful with every row of data when dealing with r?
        
    *   Can we know how good is r = 0.34 or r = -0.2 without a plot?
        
    *   Don’t forget to get the data format right for HF
        
*   HF train-validation split
    
    *   How to do the random split with HF?
        
    *   Will Jeremy talk about proper split in another notebook?
        
*   Training a model
    
    *   What to use for training a model in HF?
        
    *   What is batch and batch size?
        
    *   How large should a batch size be?
        
    *   How to find a good learning rate? (details in a future lecture)
        
    *   Where to prepare all the training arguments in HF library?
        
    *   Which type of tasks do we use for picking the model?
        
    *   How to create the learner or trainer after model?
        
    *   How to train?
        
    *   Why is the result on metrics so good right from the first epoch?
        
*   Dealing with outliers
    
    *   Should we get a second analysis for the outliers rather than simply removing them?
        
    *   What outliers really are in the real world?
        
    *   Doesn’t outliers usually tell us a lot of surprisingly useful info than in the limiting statistical sense?
        
    *   What is Jeremy’s advice on outliers?
        
*   Predict and submit
    
    *   How to do prediction with HF?
        
    *   Should we always check the prediction output as well as the test set input?
        
    *   What is the common problem with the output? (proper solution may be in the next lecture)
        
    *   What is the easy solution?
        
    *   How to submit your answer to Kaggle?
        
*   Huge opportunities in research and business
    
*   Misuses of NLP
    
    *   Can NLP chatbots can create 99% of online chat which almost non-distinguishable from real humans?
        
    *   Can GTP-3 create even longer and more sophisticated prose which is even more human-like?
        
    *   How machined generated public opinions can influence public policies or laws?
        
*   Issues on num\_labels in HF library
