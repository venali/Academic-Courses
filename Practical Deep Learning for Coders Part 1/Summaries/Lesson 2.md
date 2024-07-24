Lesson 2
========

_Daniel 深度碎片_ on [forums.fast.ai](https://forums.fast.ai/) has been kind enough to create summaries, in the form of a list of questions, of every lesson. You can use these summaries to remind yourself what you learned in each lesson, or to preview a lesson before you watch it. Here’s the lesson 2 summary:

*   New exciting content to come
    
    *   Can there be substantial new content given we have already 4 versions and a book?
        
*   Ways of reading the book
    
    *   How many channels available for us to read the book? (physical, github, colab and others)
        
*   Extra sweets from the book
    
    *   Are there interesting materials/stories covered by the book not the lecture?
        
    *   Where can you find questionnaires and quizzes of the lectures?
        
*   aiquizzes.com
    
    *   Where can you get more quizzes of fastai and memorize them forever?
        
*   Introducing the forum
    
    *   How to make the most out of fastai forum?
        
*   Students’ works after week 1
    
*   A Wow moment
    
    *   Will we learn to put model in production today?
        
*   Find a problem and some data
    
    *   What is the first step before building a model?
        
*   Access to the magics of Jupyter notebook
    
    *   Do you want to navigate the notebook with a TOC?
        
    *   How about collapsable sections?
        
    *   How about moving between start and end of sections fast?
        
    *   How to install jupyter extensions
        
*   Download and clean your data
    
    *   Why use ggd rather than bing for searching and downloading images?
        
    *   How to clean/remove broken images?
        
*   Get to docs quickly
    
    *   How to get basic info, source code, full docs on fastai codes quickly?
        
*   Resize your data before training
    
    *   How can you specify the resize options to your data?
        
    *   Why should we always use RandomResizedCrop and aug\_transforms together?
        
    *   How RandomResizedCrop and aug\_transforms differ?
        
*   Data images instantly transformed not copied
    
    *   When resized, are we making many copies of the image?
        
*   More epochs for fancy resize
    
    *   How many epochs do we usually go when using RandomResizedCrop and aug\_transforms?
        
*   Confusion matrix: where do models get wrong the most?
    
    *   How to create confusion matrix on your model performance?
        
    *   When to use confusion matrix? (category)-practice
        
    *   How to interpret confusion matrix?
        
    *   What is the most obvious thing does it tell us?
        
    *   How hard is it to tell grizzly and black bears apart?
        
*   Check out images with worse predictions
    
    *   Do plot\_top\_losses give us the images with highest losses?
        
    *   Are those images merely ones the model made confidently wrong prediction?-practice
        
    *   Do those images include ones that the model made right prediction unconfidently?
        
    *   What does looking at those high loss images help? (get expert examination or simple data cleaning)
        
*   What if you want to clean the data a little
    
    *   How to display and make cleaning choices on each of those top loss images in each data folder?-practice
        
    *   Without expert knowledge on telling apart grizzly and black bears, at least we can clean images which mess up teddy bears.
        
*   Myth breaker: train model and then clean data
    
    *   How can training the model help us see the problem of dataset?-practice
        
    *   Won’t we have more ideas to improve the dataset once we spot the problems of the dataset?
        
*   Turn off GPU when not using
    
    *   How to use GPU RAM locally without much trouble?
        
*   Watch first, then watch and code along
    
    *   What is the preferred way of lecture watching and coding by majority of students?
        
*   A Gradio + hugging face tutorial
    
*   Git and Github desk
    
    *   Is Github desk a less cool but easier and more robust way to version control than git?
        
*   Terminal for windows
    
    *   How to set up terminal for windows?
        
    *   Why Jeremy prefer windows than mac?
        
*   Get started with Hugging Face Spaces
    
    *   go to huggingface.co/spaces and create a new space
        
*   Get the default App up and running
    
    *   How to use git to download your space folder?
        
    *   How to open vscode to add app.py file?
        
    *   How to use vscode to push your space folder up to hugging face spaces online?
        
    *   then go back to your space on Hugging Face to see the app running
        
*   Train and download your model
    
    *   Where is the model we are going to train and download from Kaggle notebook?
        
    *   How to export your model after trained it on Kaggle?
        
    *   Where do you download the model?
        
    *   How to open a folder in terminal? open .
        
    *   Make sure the model is downloaded into its own Hugging Face Space folder
        
*   Predict with loaded model
    
    *   How to load downloaded model to make prediction?
        
    *   How to make prediction with the loaded model?
        
    *   How to export selected cells of a jupyter notebook into a python file?
        
    *   How to see how long a code runs in a jupyter cell?
        
*   Turn your model into Gradio App locally
    
    *   How to prepare your prediction result into a form gradio prefers? #code
        
    *   How to build a gradio interface for your model?
        
    *   How to launch your app with the model locally?
        
    *   Not in video: run the code on Kaggle in cloud
        
*   Push this app onto Hugging Face Spaces
    
    *   Make sure to create a new space first, e.g., testing
        
    *   How to turn the notebook into a python script?
        
    *   How to push the folder up to github and run app in cloud?
        
    *   Not in Video: if stuck, check out Tanishq tutorial-shooting
        
*   How many epochs are ideal for fine tuning?
    
*   How to save model from colab?
    
*   How to install fastai properly
    
    *   How to download github/fastai/fastsetup using git? git clone https://github.com/fastai/fastsetup.git
        
    *   How to download and install mamba? ./setup\_conda.sh
        
    *   Not in Video: problem of running ./setup\_conda.sh
        
    *   How to download and install fastai? mamba install -c fastchan fastai
        
    *   How to install nbdev? mamba install -c fastchan nbdev
        
    *   How to start to use jupyter notebook? jupyter notebook --no-browser
        
    *   Not in Video: other problem related to xcode
        
*   The workflow summary
    
*   HuggingFace API + gradio + Javascript = real APP
    
*   How easy does HuggingFace API work
    
*   How easy to to get started with JS + HF API + gradio
    
*   App example of having multiple inputs and outputs
    
*   App example of combining two models
    
*   How to turn your model into your own web App with fastpages
    
*   How to fork a public fastpages for your own use
