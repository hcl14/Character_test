# Training character LoRA on synthetic data

This is a short research assignment with the aim to generate synthetic data and train Stable Diffusion LoRA on them.


## Face swapping

I propose to start with cloning a certain face swapper model like [https://github.com/ai-forever/ghost](https://github.com/ai-forever/ghost) (My [fork](https://github.com/hcl14/ghost)).

Clone this repo. It is quite old, so their Colab does not work, as well as requirements.txt are ill-written.

I tested it in conda environment like this, following CUDA import errors. I supposed that I need CUDA 11.x, so I choose from the configuration from [Old pytorch versions](https://pytorch.org/get-started/previous-versions/) :

```
conda create -n swap_env pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 cudnn -c pytorch
```
Then, by trial and error, I propose this `requirements.txt`:

```
numpy==1.22
opencv-python
onnxruntime-gpu
mxnet-cu113
scikit-image
insightface==0.2.1
requests==2.25.1
kornia==0.5.4
dill
wandb
```

Please download weights using the authors' script. If there are problems, I've created [my local copy on Google drive](https://drive.google.com/drive/folders/1e2MXrnsdRoLMVVB0bf8Oq9e3kpS5KMQP?usp=sharing).


Use this source face image: [img/5-10.png](img/5-10.png) (Or you can use your own one or generated one).



You should be able to inference swap with command like

```
python inference.py --image_to_image True --source_paths='5-10.png' --target_image='TaylorSwift28.jpeg' --out_image_name='out.png'
```


At first,

**Please use face swapping functions from this repo to write python code which creates a dataset by swapping all the faces in "Taylor Swift" folder with out target face**. You can filter the ones which look broken or unrealistic.

You will have something like this:

![](img/swapped.jpg)
