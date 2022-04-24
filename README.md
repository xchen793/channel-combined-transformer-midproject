# midproject

## 1. Attention-seq2seq

#### Requirements
python3.8<br>
pytorch==1.7.1 <br>
matplotlib==3.2.2
***
#### Use
**1.train**<br/>
Just execute "trainscipt.py"<br>
**2.test**<br>
run the file "test.py"<br>
You can also modify the following code which is in the “test.py” to implement different inputs:<br>
```python
evaluateAndShowAttention("elle a cinq ans de moins que moi .")
evaluateAndShowAttention("elle est trop petit .")
evaluateAndShowAttention("je ne crains pas de mourir .")
evaluateAndShowAttention("c est un jeune directeur plein de talent .")
```
***
#### Models
saved in 'models' folder as .pkl file
***
#### Data
'data' folder
***
#### Results
1. train loss <br>
![img.png](results/img.png)
2. attention matrix <br>
![img_1.png](results/img_1.png)![img_2.png](results/img_2.png)


## 2. Transformer
#### Requirements
python3.8+<br>
pytorch==1.10+ <br>
torchtext== 0.11+ <br>
***
#### Use
**1.train**<br/>
Just execute "En_De_Ch_Att_Seq2Seq.py"<br>
**2.test**<br>
Run the file "En_De_Ch_Att_Seq2Seq.py"<br>
***
#### Results
1. accuracy
2. loss
3. attention visualization
4. LR 

## 3. Communication Channel Model
#### Use
see github: https://github.com/yihanjiang/Sequential-RNN-Decoder
