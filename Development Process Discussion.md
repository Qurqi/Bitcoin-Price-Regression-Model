## **Model Discussion:**

I used TensorFlow’s keras library for programming simplicity. Keras provides lots of
built in options for ML beginners, while still allowing for lots of customization when
needed. Also, the documentation is quite user friendly. As for data, I used a 10 year 
bitcoin price dataset from Kaggle. I have linked it rather than uploading it to this repo
in the interest of preserving Github's server storage capacity.

## **Data Discussion:**

I converted my minutely data to weekly data for improved efficiency of training
during the prototyping phase. Then, reverted to hourly data after confirming that the
model architecture ran without errors. Originally I used an arithmetic mean to aggregate 
the data.I use the geometric mean a lot in Circuit analysis, so I researched what the 
difference between geometric mean and arithmetic mean is. It turns out that geometric 
mean is better for dependent time series. Swapping to geometric mean decreased the model's 
error by about 0.5 E-1. I had to get rid of zeros to evaluate the geometric mean properly. 
I tried replacing zeros with 1’s and replacing zeros with NaN’s, then ignoring the NaN’s. 
I decided that the “1 method” logically sounded better because it accounts for the 0 value 
existence when taking the n-th root of the product, like an arithmetic mean dividing by n and
accounting for zero values. The prediction of volume was far off, but it didn’t seem to
affect the models accuracy. In hindsight, I may have been able to drop it. Without
testing to confirm this, there is really no saying what would happen though.
For the train and test data I had to create time sequences for the model to process.
I chose to use 7 hour windows to predict the 8th hour in a sequence. This choice was
somewhat arbitrary but it worked well. I are keen to experiment with this parameter in
the future. Initially windowing presented model layer I/O dimension errors, but I
realized that an option called “return_sequences” had to be enabled for all but the last
gru layer in order to process my data sequences properly. The last layer doesn’t require
a sequence because it is a deep layer.

## **Error/Loss Function Discussion:
**
I chose my error function to maximize outlier sensitivity. MSE and MSLE seemed to
be my best options after some research(see [1],[2]). Root mean squared logarithmic
error would have been ideal but after attempting to implement a custom version into
keras, I decided to stick with MSLE for programming simplicity. MSLE has better
outlier detection than MSE, making the model more robust in theory. This was proven in
practice as well. I began seeing NaN in my error function as I began to do more
fitting. This did not happen with MSE. After some research, I realized that the
logarithm part of my MSLE error function was getting a zero value. Apparently this has
to do with ReLu returning zero. The solution suggested was to add a bias. I
implemented this by changing the normalization window from (0,1) to (1,3). I chose
(1,3) instead of (1,2) because I thought that (1,3) would be less likely to saturate than
(1,2), and potentially provide better temporal pattern granularity. This change fixed my
NaN problem, allowing us to reduce model error by training more.

## **GRU Hyperparameter discussion:
**
Originally, my model consisted of three 50-node GRU layers, each with a corresponding
20% dropout layer, connected to a deep layer. These parameters were an arbitrary
choice. I discovered that keras has a built-in hyperparameter optimizer called
keras-tuner. I read the documentation on it and built a tuner that refined my layer
count, node count, dropout percentage, and adam optimizer learning rate. my
parameter space was:

● 90 < Num nodes < 200, with a step of 1

● 1E-4 < learning rate < 1E-2, no step parameter required

● 1 < Num layers < 4, with a step of 1

● 0.2 < Dropout value < 0.6, with a step of 0.05

I used BayesianOptimization to do a search of my defined parameter space because
it is a more efficient and “smart” algorithm than grid search and other alternatives [3].
I began with a broad search in the range defined above and ran the optimizer
multiple times, recording the parameter values of the highest scoring models each time.
I used the recorded data to narrow my search space, and reduce my training time
from 1 hour, 21 minutes to 21 minutes. According to the tuner results 1 layer, 137
nodes, dropout of 0.34, and learning rate of approximately 1.1E-4 was optimal for hourly
prediction.


## **References**:

[1] Jadon, A., Patil, A., & Jadon, S. (2022, November 5). A comprehensive survey of
regression based loss functions for time series forecasting. arXiv.org.
https://arxiv.org/abs/2211.02989

[2] Kapronczay, M. (2023, April 3). Mean squared error (MSE) vs. mean squared
logarithmic error (MSLE). Mean Squared Error (MSE) vs. Mean Squared Logarithmic
Error (MSLE): A Guide. https://builtin.com/data-science/msle-vs-mse

[3] Siripurapu, A., & Berman, N. (2024, January 17). Cryptocurrencies, digital dollars,
and the future of money. Council on Foreign Relations.
https://www.cfr.org/backgrounder/crypto-question-bitcoin-digital-dollars-and-future-mone
y#:~:text=Different%20currencies%20have%20different%20appeals,transaction%20or%
20charge%20a%20fee.
