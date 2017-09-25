# dqn-trader

This is still a work in progress though some parts are semi usable at this point. Call it Alpha stage if you like. The dqn is not yet implemented but the plan's to do an official release once that bit is nailed down and tuned.

You can use the env builder if you like I've listed down the steps to replicate below

## Getting data
We first need to grab a bunch of data, the scripts I have here are for open, close, high, low, adjclose, and volume, for the snp500 but you can always change things up if you want to.

To do that I shall make use of ```pandas_datareader``` to scrape snp500 data from yahoo finance. I'm also using the time frame 1/1/2005 to 16/9/2017.

Make sure that you have fulfilled these prerequisites:
- python3 (I'm using 3.6)
- numpy
- pandas
- pandas_datareader
- matplotlib

From your directory of preference do,
```
$ git clone https://github.com/mingruimingrui/dqn-trader.git
$ mkdir data
$ python src/download_data.py
```
It should take quite a while to download all that data, alternatively, drop me a message (only if you know me) I'll send you the file, it's not that large.

> I should have downloaded the stock split data and did the split transformation here but right now. If anyone wants to help, please do so. I'm putting this task on hold for now.

Next up is ```preprocessing.py``` and ```data_transform.py```. I split the two of them up into two files as they are meant to do totally different things.

We do ```preprocessing.py``` to transform the original data frame object into a multi-dimensional array of the form (timestamp, sym, feature). The advantages of doing this is three-folds,
1. Fast and easy retrieval of data (8-10x faster from local testing)
2. Memory saving (almost 2x)
3. Compatibility with pkl and npz files which makes the saving and reading process much quicker
4. (greater GPU compatibility if you are into those kinda stuff)

Then ```data_transform.py``` is more for feature engineering though I'm only using it to add in the market asset and handle stock splits at this point of time.

> The way I'm handling stock splits is not good it is best done in ```download_data.py``` as it maximizes speed

To do all that just do these two from the directory where ```main.py``` is,
```
$ python src/preprocessing.py
$ python src/data_transform.py
```

Btw all your data is stored in ```data/``` so refer to that place if you want to.

## Using env.py
Refer to ```main.py``` and ```env.py```.

In ```env.py```, there is a class called ```Env``` which is your environment maker. To create your own trading environment, you just have to call ```env = Env(timestamps, syms, col_names, data)``` as the bare minimum. You need to tell your environment the stock prices and, since we are storing the data in a multi-dimensional array, the time stamps, asset symbols, and data col_names as well.

Your ```env``` will then have a few key attributes as well as functions to use, lets go through them.

| Attr/function name        | Explaination                                                                |
| ------------------------- | --------------------------------------------------------------------------- |
| ```env.timestamps```      | list all all time stamps                                                    |
| ```env.syms```            | list of all syms in your portfolio                                          |
| ```env.col_names```       | list of all col_names                                                       |
| ```env.lookback```        | look back period                                                            |
| ```env.cur_time```        | current time frame in environment                                           |
| ```env.next_time```       | next time frame in environment                                              |
| ```action```              | action is the list of weights for each of the assets in your portfolio      |
| ```env.action_shape```    | shape of your action, typically it should be ```(len(env.syms),)```         |
| ```env.random_action()``` | get a random_action                                                         |
| ```env.step(action)```    | get to the next time frame, also returns a bunch of stuff (explained later) |
| ```env.reset()```         | reset your environment                                                      |

Actually ```action``` is not an env attribute or function, it is just a vector like object with the shape of ```env.action_shape```.

The idea is to call ```env.step``` until you reach the last time step where you can visualize your model results. Now lets go through what ```env.step``` does.

> ```state, time, reward, done = env.step(action)```

When ```env.step(action)``` is called, 4 variables are returned,
- ```state``` is the current price changes of all random assets for the look back period is of the shape (look back, no. of assets, no. of feature cols)
- ```time``` is the time frame for the states
- ```reward``` is the multiplication factor for your change in total invested value (you can change it to price change in each asset if you want to)
- ```done``` is a Boolean indicating if we have reached the last time frame

Typically we use a while look to check if ```done``` is ```True```. Once you hit done == True, then it is time to print out your results! Over in ```main.py``` you can see an example of how it is used. Have fun coding!
