Lesson 5
========

Linear model and neuralnet from scratch
---------------------------------------

*   Tabular model from scratch
    
*   Review Titanic dataset and the two models in excel
    
*   From excel to python
    
*   Clean version of notebook
    
    *   What does a clean version of the notebook look like?
        
*   Get comfortable in Paperspace Gradient
    
    *   How to work with jupyterlab mode instead of default mode?
        
    *   How to swift between jupyterlab mode and jupyter notebook mode?
        
    *   Learn some useful keyboard shortcuts
        
*   Things to do with clean notebook
    
    *   What are the steps or things we should do with the clean version of a course notebook?
        
    *   Where is the non-clean version?
        
*   Same notebook runs on Kaggle and everywhere
    
    *   How to check whether the notebook is running on Kaggle or elsewhere?
        
    *   How to get the data and its path right accordingly?
        
*   Libraries and format setup
    
    *   How much should we know about these libraries before starting?
        
    *   How to make the printed result look nicer in cells?
        
*   Read train.csv as Dataframe
    
    *   How to read and display a csv file in pandas dataframe format?
        
*   Find and count missing data with pandas
    
    *   How to check missing data in each cell/row?
        
    *   How to sum up missing data in each column?
        
*   Choose mode value for the missing data
    
    *   What is the most common choice for replacing the missing data regardless categorical or continuous? mode
        
    *   How to select the first mode value if there are two modes available for a column?
        
*   Be proactively curious
    
    *   Why it is impractical for Jeremy to explain every common function of every library used?
        
    *   What should you do about it?
        
*   Replace missing data with mode values
    
    *   How to fill in the missing data with mode values of those columns with or without creating a new dataframe?
        
*   Keep things simple where we can
    
    *   Why use the world’s simplest way of filling missing data?
        
    *   Does this simplest way work most of the time?
        
    *   Do we always know the complicated way would help?
        
*   Don’t throw out rows nor columns
    
    *   Do those filled columns sometimes turn out to matter much for the model?
        
    *   How does fastai library help to find out about it?
        
*   Describe your data or columns
    
    *   How to get a quick overview/description of your data?
        
    *   What do we look for in the descriptions?
        
*   See your columns in histogram
    
    *   What to do with interesting columns?
        
    *   What can you find out with histogram?
        
    *   What is long-tailed distribution of the column? What does it look like?
        
*   Log transformation on long-tailed columns
    
    *   Which models don’t like long-tailed distributions in the data? #data-describing
        
    *   What is the easiest way to turn long-tailed to centered distribution?
        
    *   Where to find more about the log and log curve?
        
    *   What does log do in one sentence? 17:11
        
    *   How to avoid the problem of log(0)? adding 1
        
    *   What does the column data (histogram) look like after transformed by log?
        
*   Most likely long-tailed data
    
    *   What kind of data are most likely to be long-tailed which need log transformation?
        
*   Describe non-numerical columns
    
    *   How to describe seemingly numerical but actual categorical columns?
        
    *   How to describe all non-numeric columns altogether?
        
    *   What does this description look like? (how it differ from those of numeric data)
        
*   Apply coefficients on categorical columns
    
    *   How to apply coefficients to categorical columns?
        
    *   What does it mean by applying dummy variables to categorical columns?
        
    *   What are the two ways of getting dummy variables and what’s Jeremy view on them?
        
    *   What does the dummy variable transformation of categorical variables look like ?
        
*   The secret power of name column
    
    *   Can a model built only on name column score No.1 in Titanic competition?
        
    *   Where to find more about it?
        
    *   This technique is not covered in this lecture
        
*   Tensor
    
    *   Why focus on pytorch rather than numpy?
        
    *   What data format does pytorch require? How to do this data format conversion?
        
    *   What is a tensor? Where did it come from?
        
    *   How to turn all independent columns into a single large tensor?
        
    *   What is the number type does tensor need? float
        
    *   How to check the shape of a tensor? (num of rows and columns)
        
    *   How to check the rank/dimensions/axis of a tensor? What is rank?
        
    *   What is the rank of a vector, a table/matrix, or a zero?
        
*   Create random coefficients
    
    *   Why we don’t need a constant here as in excel?
        
    *   How many coefficients we need? How we figure it out?
        
    *   How to create a vector of randomized numbers for the coefficients?
        
    *   How to make the coefficients value centered? Why this is important? (answered later)
        
*   Reproducibility of coefficients
    
    *   How to create the same set of random numbers for your coefficients each time running the cell?
        
    *   When to and not to use random seed to make your result reproducible?
        
    *   How not using random seed can help understand your model intuitively?
        
*   Broadcasting: data \* coefficients operation on GPU
    
    *   What is broadcasting? Isn’t it just matrix and vector multiplication?
        
    *   Where did it come from?
        
    *   What are the benefits of using broadcasting?
        
    *   simple code vs lots of boilerplate
        
    *   coded and optimized in C language for GPU computation
        
    *   What’s the rule of broadcasting and where to find more about it?
        
    *   a great blog post help understand broadcasting
        
*   Normalization: make the same range of values for each column
    
    *   What would happen when the values of a column is much larger than the values of other columns?
        
    *   Why to make every data column to have the same range of values?
        
    *   How to achieve the same range for all column values?
        
    *   What are the two major ways of doing normalization?
        
    *   Does Jeremy favor one over the other?
        
*   Sum up to get predictions
    
    *   How to sum up the multiplication of each row with the coefficients, and do it for all rows?
        
    *   Is the summed-up number the prediction for each person/row of data?
        
*   A default choice for loss function
    
    *   How to make the model better? Gradient descent
        
    *   What is needed to do gradient descent? loss function
        
    *   What does a loss function do? measure the performance of coefficients
        
    *   What is Jeremy’s default/favorite choice for loss function?
        
    *   Why does Jeremy always write the loss function out manually when experimenting?
        
*   Make notebook readable/understandable in the future
    
    *   When to encapsulate all exploratory steps into a few functions?
        
    *   Why keep all these exploratory steps available (Don’t delete them)?
        
*   Update coefficients with gradient descent in Pytorch
    
    *   How to ask PyTorch to do gradients on coefficients?
        
    *   How to ask Pytorch to update values on the same coefficients tensor (not create new one)?
        
    *   What does loss function do besides giving us a loss value? What does it store?
        
    *   What function to run with loss to calculate gradients for coefficients?
        
    *   How to access the gradients of coefficients? and how to interpret the gradients?
        
    *   How to decide whether it is to subtract or add gradients to coefficient?
        
    *   How to choose on the learning rate?
        
    *   How to calc updated loss with renewed coefficients?
        
*   Split the dataset
    
    *   Why did Jeremy randomly split training and validation set for Titanic dataset?
        
    *   Why to use fastai’s random splitter function?
        
    *   How to create the training and validation set with the splitter function?
        
*   Encapsulate functions for model training
    
    *   How does Jeremy create functions like init\_coeffs, update\_coeffs, one\_epoch, train\_model from the exploratory steps above?
        
    *   How to use the train\_model function to see how well the model works?
        
*   Titanic dataset is a good playground
    
    *   Why so?
        
*   Display coefficients
    
    *   How to display the final coefficients?
        
    *   How to interpret it?
        
    *   Can we make some sense of the values inside?
        
*   Accuracy as metrics
    
    *   Why not use accuracy as loss function?
        
    *   What can we use accuracy function for?
        
    *   What threshold did Jeremy use for survival?
        
    *   How to calculate accuracy and put it into a function?
        
*   Sigmoid function: ease coefficients optimization
    
    *   What you see from the predictions make you think of using sigmoid function?
        
    *   Why sigmoid function can really make optimization easier for the model?
        
    *   Why the two-ends plateau of the function is good for optimization? (to tolerate very large and small values of predictions rather than forcing every prediction to get closer to 1 or 0)
        
    *   Why the straight-line middle part of the function plot is also what we want? 48:58
        
    *   How to plot any function with just one line of code? What library is this? sympy
        
    *   How to update calc\_preds function with sigmoid function easily in Jupyter? 50:52
        
    *   Why to make predictions to center on 0 before sigmoid function? (a reply by Jeremy)
        
    *   Do you remember what did Jeremy do to make prediction center on 0? (see how initial coefficients is defined, a cell link on Kaggle)
        
    *   Why allow predictions to be large or small can make weights optimization easier? (Jeremy’s reply)
        
    *   How python with Jupyter make exploratory work so easy?
        
    *   How come the learning rate jump from 0.1 before sigmoid to 2 after using sigmoid? 51:57
        
    *   When or How often (as a rule) should you use sigmoid function to your prediction? 52:23
        
    *   Does HF library specify whether they use sigmoid or not? (probably the others neither)
        
    *   What You need to watch out for optimization is the input and output not the middle for now. Why is that? 53:13
        
*   What if test dataset has extra columns?
    
    *   What would the normal consequences be?
        
    *   How does fastai deal with it for good?
        
*   Submit to Kaggle
    
    *   How and why Jeremy replaced a missing value of Fare with 0?
        
    *   How to apply the above data cleaning steps to the test set?
        
    *   How to prepare the output column expected by Kaggle?
        
    *   How to create the submit file expected by Kaggle?
        

Key steps from linear model to neuralnet
----------------------------------------

*   val\_indep \* coeffs vs val\_indep @ coeffs
    
    *   What do we know about val\_indep \* coeffs mean? Is it element-wise? Is it matrix and vector multiplication?
        
    *   What do we know about val\_indep @ coeffs? Is it matrix-matrix multiplication?
        
    *   Is (val\_indep \* coeffs).sum(axis=1) equal to val\_indep @ coeffs?
        
    *   What should we know about them to distinguish them properly?
        
    *   In val\_indep @ coeffs, when coeffs is a matrix, do we need to specify its shape? 59:50
        
    *   How to initiate coefficients as a one column matrix rather than a vector?
        
*   Transform existing vectors into matrix
    
    *   How to turn both trns\_dep and vald\_dep from existing vectors to matrices which responding to coeffs matrix?
        

Building a neural net
---------------------

*   Keep linear model and neuralnet comparable in output
    
    *   How to create a layer within multi-hidden layers inside (or a matrix of coefficients rather than a vector of coefficients)?
        
    *   why to divide the coefficients of the multi-hidden layers by the number of layers (or the matrix of coefficients by the number of columns)?
        
    *   Is it to make sure the outputs of neuralnet and previous linear model are comparable?
        
*   Build the output layer
    
    *   How to build the coeffs of the output layer with correct shape which connects with the previous layer?
        
    *   How to decide the number of output of this output layer?
        
*   TRY to getting the training started
    
    *   Why Jeremy make the coefficients of the output layer to minus 0.3?
        
    *   What does it mean by this minus 0.3 can get the training start?
        
    *   (I guess Jeremy may tried -0.5 first, experiment to find it out)
        
*   Adding Constant or not
    
    *   Why we don’t need a constant for layer 1 (think of the constant of the linear model)?
        
    *   Why we must have a constant for layer 2?
        
    *   Do coefficients of layer1, layer 2 and constants all need their own gradients initiated?
        
*   Building the model
    
    *   What is a tuple and how it is used for grouping and separating the three coefficients?
        
    *   How to construct the prediction function by sending data through layer 1 and layer 2 finally add constants?
        
*   A neuralnet done but super fiddly
    
    *   How to update all three coefficients in a loop?
        
    *   Did you notice that the learning rate changed again? (1.4, last time was 2, earlier was 0.1)
        
    *   What did Jeremy say getting this model work was super fiddly?
        

From neuralnet (1 hidden layer) to deep learning with 2 hidden layers
---------------------------------------------------------------------

*   Initialize coefficients of all hidden layers and constants
    
    *   How to initialize coefficients of 2 hidden layers and 1 output layer and constants, and get gradients ready for all of them, in one compact function?
        
    *   What are the shape of each coefficient matrix?
        
*   Building the 2 hidden layer model
    
    *   What are activation functions?
        
    *   What are the activation functions for 2 hidden layers?
        
    *   What is the activation function for the output layer?
        
    *   What is the most common mistake on applying activation function to the final layer?
        
*   Train the model
    
    *   Don’t forget to update gradients
        
    *   Which are those numbers Jeremy still have to fiddle to get right?
        
    *   Did this deep learning model improve on the loss and accuracy?
        
*   Dissect and Experiment large functions
    
    *   How to experiment on a large function like the init\_coeffs by breaking it into small pieces and running them?
        
*   Tabular datasets: where deep learning not shining
    
    *   How should we think about that both neuralnet and deep learning models aren’t better?
        
    *   What does it mean that a carefully designed algo beat all deep learning models in Titanic competition?
        
    *   What datasets do deep learning generally perform better?
        

Framework: DL without super fiddling notebook
---------------------------------------------

*   Why use framework rather than from scratch
    
    *   Why you should use a library framework in real life rather than building yourself like the above?
        
    *   When to do it from scratch?
        
    *   What can a framework do for us?
        
    *   Can it automate the obvious things like initialization, learning rate, dummy variable, normalization, etc?
        
    *   Can I still make choice on the not-so obvious things?
        
*   Feature engineering with pandas
    
    *   What is the feature engineering with pandas look like?
        
    *   How and where does Jeremy suggest on digging in pandas?
        
*   Automate the obvious when preparing dataset
    
    *   How framework make cateorifying data, filling missing data, normalization automatic?
        
    *   How to specify the dependent column to be a category?
        
*   Build multi-hidden layers with one line of code
    
    *   How to specify the shapes of two hidden layers with just two numbers?
        
    *   Do you only need to specify accuray without worrying about loss and activations?
        
*   Automate the search of best learning rate
    
    *   How does fastai help you find the range where best learning rate locates?
        
    *   How should you pick the learning rate from the range?
        

Predict and Submit with ease
----------------------------

*   Automate transformation of test set in one line of code
    
    *   How to automatically apply all transformations done to training and validation sets to test set?
        

Experiment with Ensembling
--------------------------

*   Ensemble is easy with fastai
    
    *   Does framework save so much of fiddling so that experimenting with some advanced ideas become easier?
        
    *   What is ensembling?
        
    *   Is it to combine multiple models and combine their predictions?
        
*   The simplest ensemble
    
    *   What does a simple ensemble look like?
        
    *   How to build, run and predict with 5 models with ease?
        
    *   How different are those 5 models? (only initial coefficients are different)
        
    *   How to combine their predictions?
        
    *   How much improvement does this simple ensemble get us to?
        
*   Ways to combine the predictions
    
    *   Why not use mode but mean?
        
    *   What are 3 ways of combining the predictions?
        
    *   Does one is better than the others?
        
    *   What’s Jeremy’s suggestion?
        

How Random Forest really work
-----------------------------

*   Is this a good place to also learn pandas and numpy?
    
*   Why Random Forest
    
    *   What is the history of Random Forest and Jeremy?
        
    *   What does Jeremy think of random forest?
        
    *   Why random forest is so much easy and better?
        
    *   Why the seemingly simple logistic regression is so easy to get wrong?
        
*   Pandas categorical function
    
    *   How to import all the libraries you need at once?
        
    *   How to do fillna with pandas and log with numpy?
        
    *   What does panda categorical function do for us?
        
    *   What’s friendly display after the function applied?
        
    *   What’s the actual data transformation under the hood?
        
    *   Key points to make: No dummy variables, Pclass no long needed to be viewed as categories
        
*   Binary splits: bases of random forest
    
*   A binary splits on gender
    
*   Build a binary splits model on gender with sklearn
    
*   Build a binary splits model on Ticket prices with sklearn
    
*   Build a score machine on binary splits regardless categorical or continuous
    
    *   What is a good split?
        
    *   Is it good that within each group their dependent values are similar?
        
    *   How to measure the similarity of values within a group? std
        
    *   How to compare standard deviations between two groups appropriately? (multiply by size)
        
    *   How to calc the score for evaluating the splits based on the value of combined std of two groups?
        
*   Automate the score machine on all columns
    
    *   How to find the best binary splits by trying out all possible split points of a column?
        
*   1R model as the baseline
    
    *   What is a random forest? and what is a random forest?
        
    *   What is 1r model?
        
    *   how good was it in the 90s of ML world?
        
    *   Should we always go for complicated models?
        
    *   Should we always start with a 1r model as a baseline model for our problem?
