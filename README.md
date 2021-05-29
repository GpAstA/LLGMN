# R-LLGMN for PyTorch
By Yuki Hashimoto

## [A log-linearized Gaussian mixture network and its application to EEG pattern classification](https://ieeexplore.ieee.org/abstract/document/740670)
You should read [this journal](https://ieeexplore.ieee.org/abstract/document/740670) before you use this program.<br>

## About the program
LLGMN layer for PyTorch.<br>

## Getting Started 
Clone repository:
```
git clone
```
Install [PyTorch](https://pytorch.org/get-started/locally/) and dependencies:
```
pip3 install pytorch    # <- Depends on your OS and python package.
pip3 install -r requirements.txt
```

## Quick Run
I prepared a very simple example code.<br>
This may help you understand how to use R-LLGMN layer.<br>
```
python main.py
```

## Notes
- Use only NLLLoss because R-LLGMN output is like softmax.
- I think processing speed would be able to be faster, because the cell needs for loop.
- Please tell me, if you find any bug.
- I checked this code work on Win10 and Ubuntu18.04. Sorry I don't have Mac.

## References
[A log-linearized Gaussian mixture network and its application to EEG pattern classification](https://ieeexplore.ieee.org/abstract/document/740670)<br>