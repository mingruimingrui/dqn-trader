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

To download this github repo and the dataset, from your directory of preference do,

```
$ git clone https://github.com/mingruimingrui/dqn-trader.git
$ mkdir data
$ python src/download_data.py
```

It should take quite a while to download all that data, alternatively, drop me a message (only if you know me) I'll send you the file, it's not that large.

The data is stored in a three dimensional matrix in the layout [ timestamp , ticker(or symbol), data_name ]. Benefits to storing data this way rather than in a traditional table like dataframe is 3 folds,

1. Increased speed of information retrieval since the space of sorting and filter is greatly reduced (8-10x faster from local testing)
2. Memory saving (almost 2x)
3. Leverage on npz/pkl file which greatly improves speed of file reading for running code on shell
4. (greater GPU compatibility if you are into those kinda stuff)

Next up is ```preprocessing.py```. The plan is to use this file for feature engineering though I'm only using it to add in the market asset and handle stock splits at this point of time.

Keep in mind, the code needs to be ran from the directory where ```main.py``` is,

```
$ python src/preprocessing.py
```

Btw all your data is stored in your created ```data``` file so refer to that place if you want to.

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
- ```state``` is the price changes of all random assets for the look back period is of the shape (look back, no. of assets, no. of feature cols) it is also normalized according to current open prices.
- ```time``` is the time frame for the states
- ```reward``` is the multiplication factor for your change in total invested value since the previous time frame (you can change it to price change in each asset if that is better)
- ```done``` is a Boolean indicating if we have reached the last time frame

As ```state``` is normalized to cur_open price, you can tell that the environment is trading at daily open. Do print out ```state``` and ```time``` to figure out exactly they are as people usually have problem with understanding them.

Typically we use a while look to check if ```done``` is ```True```. Once you hit done == True, then it is time to print out your results! Over in ```main.py``` you can see an example of how it is used. Have fun coding!
