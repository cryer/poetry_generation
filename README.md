# poetry_generation

Use LSTM to do poetry generation with pytorch.

## Inspiration
* [chenyuntc](https://github.com/chenyuntc/pytorch-book/tree/master/chapter9-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%86%99%E8%AF%97(CharRNN))

## Datasets

[chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)

But data has already converted into numpy binary files,you can find it in data directory.
This file is also provided by chenyuntc.

## Train

```
git clone https://github.com/cryer/poetry_generation.git
cd poetry_generation
python train.py
```
All config is in Config.py,such as learning rate,epoch,batchsize and so on.
feel free to modify.

## Pretrained model

I have already trained this model and put checkpoint file in checkpoints subdirectory.

Clone this repo and run code below:
```
python demo.py gen --use_gpu=False
```
All options are:(including training options)
```
    --data_path = 'data/'
    --pickle_path = 'data/tang.npz'
    --lr = 1e-3
    --use_gpu = True
    --epoch = 20
    --batch_size = 128
    --max_gen_len = 200
    --model_path = 'checkpoints/final.pth'
    --prefix_words = '庄生晓梦迷蝴蝶，望帝春心托杜鹃。'
    --start_words = '如果能重来'
    --acrostic = False
    --model_prefix = 'checkpoints/cpkt'
```
## Explaination
```
prefix_words = '庄生晓梦迷蝴蝶，望帝春心托杜鹃。'
```
* This will not appear in your generation,it just provides some artistic conception to your generation.
Actually,it just means you have to put prefix_words into your model first, the output hidden part will
then be used to put into real generation part as initial hidden input of LSTM.If you set prefix_words
to None,initial hidden input of LSTM will set zero.
* If your prefix_words is seven characters per line,then generation poetry may be likely to  be seven characters per line.
If five characters per line,so is generation.Mind that this phenomenon is learned by model itself,not man-made，we
did not set their length to that.That means it won't all the time such.

```
start_words = '如果能重来'
```
This is the start words of your generation,it has two conditions:
* If `acrostic = False` ,that means generation poetry is not acrostic,so start words will
appear in the beginning of your generation poetry.
* If `acrostic = True` ,that means generation poetry is a acrostic,so start words will be separated
to be the first word of each sentence.
 
 ## Results
 
 * acrostic = True :
 ```
 如今不见江南柳，不及南山作雨烟。果取枯桑叶落屋，叶开花里不胜蝉。能怜玩物不可把，未见余花不可怜。
 重叹昔时为有意，不如今日是谁怜。来时自有同心友，独坐相逢不可听。
 ```
  * acrostic = False :
 ```
如果能重来不得，如今不觉有花殊。佯佯刷沥不成啄，乍啄林梢生万枝。
吴樟岸苇渔竿上，荻岸洲汀烟雾波。蒲萄蔽叶纷裴度，野屿草茸生暮暮。
湖南女儿一枝鹤，一双蹙浪飞奔螭。一夕清光照空碧，万里无人鸟兽孤。
一朝相见皆悲汎，几处旌旗傍路岐。此时別后相思处，一夜清风吹一枝。
 ```
 



