Lesson 3
========

*   Introduction and survey
    
*   “Lesson 0” How to fast.ai
    
    *   Where is Lesson 0 video?
        
    *   What does it to do with the book ‘meta learning’ and fastai course?
        
*   How to do a fastai lesson?
    
    *   Watch with note
        
    *   Run the notebook and experiment
        
    *   Reproduce the notes from the codes
        
    *   Repeat with a different dataset
        
*   How to not self-study?
    
    *   physical and virtual study group
        
    *   study with people on forum
        
    *   Learn with social interactions is better than self-study
        
*   Highest voted student work
    
    *   Many interesting projects to check out
        
*   Jeremy’s Pets breeds detector
    
    *   Jeremy’s Pets repository
        
    *   What you should do with this App example?
        
*   Paperspace: your DL workstation in cloud!
    
    *   Does Jeremy speak highly of it? and Why?
        
*   JupyterLab: real beginner friendly
    
    *   Why JupyterLab is so good for beginners to take advantage of?
        
*   Make a better pet detector
    
    *   After training, we should think about how to improve it
        
*   Comparison of all (image) models
    
    *   Did anyone compared most of the image models and shared the finding?
        
    *   Where to find the notebook for comparison?
        
    *   Which 3 criteria are used for comparison?
        
*   Try out new models
    
    *   How to select and try out models with high scores
        
    *   Where is the train.ipynb file?
        
    *   How to try models on TIMM?
        
    *   How to compare them by loss?
        
    *   Why this model is actually impressive?
        
    *   What can the name of a model tell us?
        
    *   Why Jeremy only train 3 epochs? 18:58
        
*   Get the categories of a model
    
    *   How to get labels or categories info from the model?
        
    *   The rest is we learnt from last lecture.
        
*   What’s in the model
    
    *   What two things are stored in the model?
        
*   What does model architecture look like?
    
*   Parameters of a model
    
    *   How to zoom in on a layer of a model?
        
    *   How to check out the parameters of a layer?
        
    *   What does a layer’s parameters look like?
        
*   The investigating questions
    
    *   What are the weights/numbers?
        
    *   How can they figure out something important?
        
    *   Where is the notebook on how neuralnet work
        
*   Create a general quadratic function
    
    *   How to create a general function to output any specific quadratic function by changing 3 parameters?
        
    *   How to generate result from a specific quadratic function by changing 1 parameter?
        
    *   Why do we create such a general (quadratic) function with multiple unknown parameters rather than directly writing a particular quadratic function with specific coefficients?
        
*   Fit a function by good hands and eyes
    
    *   What does fit a function mean? (search better parameters based on dataset)
        
    *   How to create a random dataset?
        
    *   How to fit a general quadratic function to the dataset by changing 3 parameters with jupyter widgets by hand?
        
    *   What is the limitation of this manual/visual approach?
        
    *   where is this notebook
        
*   Loss: fit a function better without good eyes
    
    *   Why do we need loss or loss function?
        
    *   What is mean squared error?
        
    *   How does loss help the hand/visual approach to be more accurate and robust?
        
*   Automate the search of parameters for better loss
    
    *   How do we know which way and by how much to update parameters in order to improve on loss?
        
    *   Can you find enough derivative material on Khanacademy?
        
    *   What exactly do you need to know about derivative for now according to Jeremy? 34:26
        
    *   What is the slope or gradient?
        
    *   Does pytorch do derivative or calc slope/gradient for us?
        
    *   How to create a function to output sme loss on a general quadratic function? 35:02
        
    *   What do you need to know about tensor related to derivatives for now according to Jeremy? 36:02
        
    *   How to create a rank 1 tensor (a list to store numbers) to store parameters of the quadratic function? 36:49
        
    *   How to ask pytorch to prepare the calculation of gradients for these parameters? 37:10
        
    *   How to actually calculate gradients for each parameter based on the loss achieved by this specific function (3 specific parameters) against the whole dataset? 37:38
        
    *   In other words, this time when we calculate loss we can easily get the gradient for each parameter as well.
        
    *   What does the gradient value mean for each parameter? 38:34
        
    *   How to update parameters into new values with the gradients produced by the loss? 39:18
        
    *   How to automate the process above to find better parameters to achieve better loss? 41:05
        
    *   Why this automation is called gradient descent?
        
    *   notebook
        
*   The mathematical functions
    
    *   Besides dataset, loss function, derivative, what is also very crucial in finding/calculating those parameters?
        
    *   Why we can’t simply use quadratic functions for it?
        
*   ReLu: Rectified linear function
    
    *   Real world powerful models demands complex parameters and also complex functions, how complex a function can we come up?
        
    *   Is it possible to come up an infinitely complex function by simply doing addition of extremely simple functions?
        
    *   What could such extremely simple function look like?
        
    *   What is rectified linear function? How simple it is? What is linear and which part is rectified?
        
    *   What does rectified linear function look like in plot?
        
    *   How to adjust the 2 parameters of the function by hand with widget?
        
    *   What the function could look like under different parameters? 44:46
        
*   Infinitely complex function
    
    *   How powerful can the addition of extremely simple functions be?
        
    *   How to create a double rectified linear function (double relu) and adjust 4 parameters by hand with widget?
        
    *   How much more flexible does this double relu function look compared to a single rectified linear function?
        
    *   Can you imagine how complex can a function be when millions of rectified linear functions are added?
        
*   2 circles to an owl
    
    *   a very concise summarization of sewing fundamental ideas together for deep learning
        
*   A chart of all image models compared
    
    *   Can it be done with brute force computation with simple code?
        
    *   Does Jeremy look for the model comparison chart for best models?
        
    *   What is the wrong way of using the comparison chart by students? 50:45
        
    *   How does Jeremy use the chart?
        
    *   how does Jeremy decides which models to try out step by step?
        
*   Do I have enough data?
    
    *   Did you already build a model and train on your own dataset?
        
    *   Is the result good enough for you?
        
    *   What is the mistake the DL industry often make on this issue? 52:55
        
    *   What is Jeremy’s suggestion?
        
    *   How and what could semi-supervised learning and data augmentation be helpful?
        
    *   What about labeled and unlabeled data?
        
*   Interpret gradients in unit?
    
    *   How much does the loss go down when parameter a increase by unit of 1? 55:24
        
*   Learning rate
    
    *   Why we don’t update parameter values in large steps?
        
    *   Why does Jeremy draw a quadratic function to refer to the model when zooming in very close into the complex function?
        
    *   What would happen when update parameters with large values? 57:19
        
    *   Does large drop on loss necessarily demand large value increase of parameter according to the quadratic nature?
        
    *   What is learning rate? Why we need it to be small? How to pick a good value of it? 58:07
        
    *   What would happen if your learning rate is too big?
        
    *   What would happen when too small?
        
*   break
    
*   Matrix multiplication
    
    *   When the model requires millions of rectified linear functions, how to calculate fast enough?
        
    *   What is actually needed from linear algebra to do DL 1:01:33
        
    *   How easy it is to do matrix multiplication? 1:01:51
        
    *   What are the dataset and parameters in the matrix multiplication?
        
    *   Does matrix multiplication do the rectified part for you?
        
    *   What GPU is good at? 1:03:49
        
*   Build a regression model in spreadsheet
    
    *   Intro to Titanic Competition on Kaggle 1:05:01
        
    *   What is the dataset 1:05:18
        
    *   What to do with the train.csv file?
        
    *   How to clean the dataset a little bit?
        
    *   How to transform the dataset for matrix multiplication? 1:07:17
        
    *   How to prepare parameters for matrix multiplication? 1:08:50
        
    *   What’s wrong with the much larger value of the column ‘Fare’ compared to other columns? 1:09:35
        
    *   What to do with the values of ‘Fare’ and similarly the values of ‘Age’?
        
    *   What is normalizing the data?
        
    *   Does fastai do all these normalizations for us? Will we learn how fastai does it in the future?
        
    *   Why to apply log to values of ‘Fare’? 1:10:59
        
    *   Why do we need values to be evenly distributed?
        
    *   How to do mmult on dataset and parameters in spreadsheet? 1:11:56
        
    *   How to use mmult instead of addition to add a constant?
        
    *   What does the result of our model look like? 1:13:41
        
    *   Does Jeremy simply use a linear regression for the model, not even a relu?
        
    *   Can we solve regression with gradient descent? How do we do it?
        
*   Build a neuralnet by adding two regression models
    
    *   What does it take to turn a regression model into a neuralnet?
        
    *   Why we don’t add up the results of two linear functions?
        
    *   Why we only add the results together after they are rectified?
        
    *   What does the model prediction look like?
        
    *   Now we need to update the parameters for two linear functions, not just one.
        
*   Matrix multiplication makes training faster
    
    *   How to make the training to do mmult rather than addition of linear multiplications in spreadsheet?
        
*   Watch out! it’s chapter 4
    
    *   Please do try out Titanic competition
        
    *   Why chapter 4 drove away most of people?
        
    *   Ways to work out the spreadsheet yourself
        
*   Create dummy variables of 3 classes
    
    *   Do we only need 2 columns/classes for a dummy variable with 3 classes?
        
*   Taste NLP
    
    *   What do Natural Language Processing models do?
        
    *   What project opportunities do non-En-speaker students have?
        
    *   What tasks can NLP do? 1:25:57
        
*   fastai NLP library vs Hugging Face library
    
    *   How do these two libraries differ?
        
    *   Why we use transformer library in this lecture?
        
*   Homework to prepare you for the next lesson
