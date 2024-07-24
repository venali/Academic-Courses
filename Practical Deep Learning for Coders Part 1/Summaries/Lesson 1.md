Lesson 1
========

_Daniel 深度碎片_ on [forums.fast.ai](https://forums.fast.ai/) has been kind enough to create summaries, in the form of a list of questions, of every lesson. You can use these summaries to remind yourself what you learned in each lesson, or to preview a lesson before you watch it. Here’s the lesson 1 summary:

*   Welcome to Part 1 2022 course
    
*   Were computers smart enough to determine photos of birds before 2015?
    
*   How to download and display a photo of a bird from DuckDuckGo using simple codes?
    
*   What photos/images are actually made of, at least for computers?
    
*   How to create two folders named ‘bird’ and ‘forest’ respectively under a larger folder ‘dest’? How to download 200 images for each category? How to resize and save those images in respective folders?
    
*   How to find broken images and then remove or unlink them from their folders?
    
*   How to create a DataBlock which prepares all the data for building models? How to display the images in a batch?
    
*   How to build a model and train/finetune it on your local computer?
    
*   How to predict or classify a photo of bird with a model?
    
*   How to get started running and playing around the codes and models immediately and effortlessly?
    
*   Why should you read lecture questionnaires before studying the lecture?
    
*   How do you search and locate a particular moment inside a lecture video?
    
*   Can you create an original masterpiece painting by simply utterring some artistic words?
    
*   Can you believe that models today can explain your math problems not just give you a correct answer? Can you believe that models today can help you get a joke?
    
*   Jeremy and fastai community make serious effort in help beginners continuously.
    
*   Do you want to know how to make the most out of fastai?
    
*   Do you know people learn naturally (better) with context rather than by theoretical curriculum? Do you want this course to make you a competent deep learning practitioner by context and practical knowledge? If you want theory from ground up, go to part 2 fastai 2019
    
*   Do you know that learning the same thing in different ways betters understanding?
    
*   Why you must take this course very seriously? (Personally, I think it’s truly a privilege to be taught by Jeremy and to be part of the fastai family. I didn’t appreciate it enough as I should 4 years ago.)
    
*   Why did we need so many scientists from different disciplines to collaborate for many years in order to design a successful model before deep learning?
    
*   Why can deep learning create a model to tell bird from forest photos in 2 minute which was the impossible before 2015? Would you like to see how much better/advanced/complex are the features discovered by deep learning than groups of interdisciplinary scientists?
    
*   Are all things are data, sound, time (series), movement? Are images are just one way of expressing data? Why not store or express data (of sound, time, movement) in the form of images? Can imaged based algos learn on those images no matter how weird they appear to humans?
    
*   Can I do DL with no math (I mean with high school math)? Can I train DL models with hand-made data (<50 samples)? Can I train state of art models for free (literally)?
    
*   Which should I invest my life in DL software field, Pytorch or Tensorflow?
    
*   Why should you use fastai over pure pytorch? Don’t you want to write less code, make less error, achieve better result? Don’t you want a robust and simple tool used by your future colleagues and bosses?
    
*   Why is jupyter notebook the most loved and tested coding tool for DL? Do you want Jeremy to show you how to use Jupyter notebook hand by hand?
    
*   How to make sure your notebook is connected in the cloud? How to make sure you are using the latest updated fastai? #best-practice
    
*   Doesn’t fastai feel like python with best practices too? How to import libraries to download images? How to create and display a thumbnail image? Always view your data at every step of building a model #best-practice How to download and resize images? Why do we resize images? #best-practice
    
*   Why a real world DL practitioner spend most of the valuable/productive time preparing data rather than tweaking models? Can super tiny amount of models solve super majority of practical problems in the world? Have fastai selected and prepared the best models for us already?
    
*   Does Jeremy add best practices of other programming languages into fastai? Jeremy loves functional programming
    
*   How fastai design team decide what tasks should DataBlock do? task 1: Which blocks of data do DataBlock need to prepare for training? task 2: How should DataBlock get those data, or by what function/tool? task 3: Should we always ask DataBlock to keep a section of data for validation? task 4: Which function or method should DataBlock use to get label for y? task 5: Which transformation should DataBlock apply to each data sample? task 6: Does dataloader do the above tasks efficiently by doing them in thousands of batches at the same time with the help of GPUs?
    
*   What is the most efficient way of finding out how to use e.g., DataBlock properly? How to learn DataBlock thoroughly?
    
*   What do you give to a learner, e.g., vision\_learner?
    
*   Is fastai the first and only framework implement TIMM? Can you use any model from TIMM in your project? Where can you learn more of TIMM?
    
*   What is a pretrained model, Resnet18? What did this model learn from? What come out of this model’s learning? or what is Kaggle downloading exactly?
    
*   What exactly does fine tuning do to the pretrained model? What does fine-tuning want the model learn from your dataset compared with the pretrained dataset?
    
*   How to use the fine tuned model to make predictions?
    
*   Can we fine tune pretrained CV models to tell us the object each and every pixel on a photo belong to?
    
*   Why do we need specialized DataLoaders like SegmentationDataLoaders given DataBlock?
    
*   What can tabular analysis do? Can we use a bunch of columns to predict another column of a table? How do you download all kinds of dataset for training easily with fastai? untar\_data What are the parameters for TabularDataLoaders? What is the best practice show\_batch of fastai learned from Julia (another popular language)? Why to use fit\_one\_cycle instead of fine\_tune for tabular dataset?
    
*   Can we use collaborative filtering to make movie recommendations for users? How does recommendation system work? Can collaborative filtering models learn from data of similar music users and recommend/predict music for new users based on how similar they are to existing users?
    
*   How to download dataset for collaborative filtering models? How to use CollabDataLoaders? How to build a collaborative filtering model with collab\_learner? What is the best practice for setting y\_range for collab\_learner? #best-practice If in theory no reason to use pretrained collab models, and fine\_tune works as good as fit or fit\_one\_cycle, any good explanations for it? #question How to show results of this recommendation model using show\_results?
    
*   What can Deep Learning do at the present? What are the tasks that deep learning may not be good at?
    
*   Has the basic idea of deep learning changed much since 1959?
    
*   What did we write into programs/models before deep learning? How to draw chart in jupyter notebook?
    
*   What is a model? What are weights? How do data, weights and model work together to produce result? Why are the initial results are no good at all? Can we design a function to tell the model how good it is doing? loss function Then can we find a way to update/improve weights by knowing how bad/good the model is learning each time from the data? If we can iterate the cycle multiple times, can we build a powerful model?
    
*   Homework: Run notebooks, especially the bird notebook. Create something interesting to you based on the bird notebook. Read the first chapter of the book. Be inspired by all the amazing student projects.
