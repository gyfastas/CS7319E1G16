## CS7319E1G16

This is a unoffical PyTorch implementation of the [DCR paper](https://ieeexplore.ieee.org/abstract/document/8303213/):
```
@Article{lu2018deep,
  title={Deep coupled resnet for low-resolution face recognition},
  author={Lu, Ze and Jiang, Xudong and Kot, Alex},
  journal={IEEE Signal Processing Letters},
  volume={25},
  number={4},
  pages={526--530},
  year={2018},
  publisher={IEEE}
}
```

### Environments

- **OS**: Ubuntu18.04.4 LTS
- **GCC/G++**: 7.5.0
- **GPU**: GeForce GTX TITAN X *4 |12G Memory
- **CUDA Version**: 10.1
- **NVIDIA-SMI**: 450.57
- **CPU**: Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz * 12.

### Installation

Install requirements:
```
pip install -r requirements.txt
```

For [faiss](https://github.com/facebookresearch/faiss), we recommend installing with the following command:

```
pip install faiss-gpu
```

See [faiss-wheels](https://github.com/kyamagu/faiss-wheels) for more details.

For [PyTorch](https://pytorch.org/) , we recommend following the instruction on official website.

### Data Folder Structure

We support training and testing on [Casia](https://jbox.sjtu.edu.cn/l/fJ6wsU), [LFW](https://jbox.sjtu.edu.cn/l/pn3mat) and [SCface](https://jbox.sjtu.edu.cn/l/b1kNW8 ). They should be renamed and placed in the same data root as:

```
Data Root/
	Casia/
		120x120_120/
		casia_cleaned_list.txt
	LFW/
		LFW_120x120_120/
		lfw_test_pair.txt
	SCface/
		mugshot_frontal_cropped_all/
		surveillance_cameras_distance_1/
		surveillance_cameras_distance_2/
		surveillance_cameras_distance_3/
```

### Reproduce DCR Trunk

This implementation supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu is also supported.

To reproduce our best result on Trunk with single-gpu:
```
python train.py --gpu 0 -j 16 --scale-list 8 12 16 20 --patience 20 --lr 1e-2 --dataset "Casia" \
--val-dataset "LFW" \
--schedule 10 30 60 \
--optimizer "Adam" \
--normalized_embeddings \
--subset 11 \
--aug-plus \
--task "V" \
[your data root folder with Casia, LFW and SCface datasets.] \
[your logging folder] \
"trunk"

```
To test on [SCface](https://www.scface.org/):

```
python test.py --dataset "SCface" \
[your data root folder with Casia, LFW and SCface datasets.] \
[config file of your trained model, your checkpoint path/config.json] \
[your checkpoint path/model_best.pth.tar] \
"trunk" \
"R"
```

To fine-tune on  [SCface](https://www.scface.org/):

```
python train.py --gpu 0 -j 16 --patience 20 --lr 1e-2 --dataset "SCface" \
--val-dataset "SCface" \
--schedule 10 30 60 \
--optimizer "Adam" \
--normalized_embeddings \
--load [your checkpoint path/model_best.pth.tar] \
--task "R" \
[your data root folder with Casia, LFW and SCface datasets.] \
[your logging folder] \
"trunk"
```



***Note***: we also support 10-fold validation on [Casia](https://pgram.com/dataset/casia-webface/), set `--subset [x]` with x smaller than 10 and `--val-dataset "Casia"` .


### Reproduce DCR Branch

With a trained Trunk, to train a branch with **specialized initialization** on  [Casia](https://pgram.com/dataset/casia-webface/) and validate on [LFW](http://vis-www.cs.umass.edu/lfw/), run:
```
python train.py --gpu 0 -j 16 --scale-list 8 12 16 20 --patience 20 --lr 1e-2 --dataset "Casia" \
--val-dataset "LFW" \
--schedule 10 30 60 \
--optimizer "Adam" \
--normalized_embeddings \
--subset 11 \
--init-branches \
--aug-plus \
--task "V" \
--resume [your checkpoint path/model_best.pth.tar] \
[your data root folder with Casia, LFW and SCface datasets.] \
[your logging folder] \
"branch"
```



***Note***: To use random initialization, remove `--init-branches`. See our report for more details.




### Models

Our pre-trained trunk and branch models can be downloaded as following (the result is verification accuracy on [LFW](http://vis-www.cs.umass.edu/lfw/) ) :
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">8x8</th>
<th valign="bottom">12x12</th>
<th valign="bottom">16x16</th>
<th valign="bottom">20x20</th>
<th valign="bottom">HR</th>
<th valign="bottom">model</th>
<th valign="bottom">passwd</th>
<!-- TABLE BODY -->
<tr><td align="left">Ours-Trunk</a></td>
<td align="center">86.9</td>
<td align="center">91.0</td>
<td align="center">93.0</td>
<td align="center">93.0</td>
<td align="center">93.5</td>
<td align="center"><a href="https://pan.baidu.com/s/1SZc15GF-YlPSNA3DrxAGcA">download</a></td>
<td align="center"><tt>ycvw</tt></td>
</tr>
<tr><td align="left">Ours-SI Branch</a></td>
<td align="center">77.7</td>
<td align="center">90.8</td>
<td align="center">90.3</td>
<td align="center">92.2</td>
<td align="center">91.5</td>
<td align="center"><a href="https://pan.baidu.com/s/1tSZ_ZF2Ppa3fmUWKkugDTA">download</a></td>
<td align="center"><tt>gdjh</tt></td>
</tr>
</tbody></table>

More results on [SCface](https://www.scface.org/) could be seen in our report.




### License

This project is under the MIT license. See [LICENSE](LICENSE) for details.



